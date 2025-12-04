import argparse
import os
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.datasets_finetune_segmentation import SentinelIndividualImageDataset
from engine_finetune_seg_e1 import train_one_epoch, evaluate
from models_mae_seg_e1 import MAEGroupChannelsSegE1

def get_args_parser():
    parser = argparse.ArgumentParser('E1: Segmentation with MAE encoder+decoder (tune decoder key parts)', add_help=False)
    parser.add_argument('--julaug_dir', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\bole_data\JulAug_croped', help='JulAug 10-band imagery root')
    parser.add_argument('--mask_dir', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\bole_data\label_post_croped', help='Segmentation masks root')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--in_chans', default=10, type=int)
    parser.add_argument('--nb_classes', default=3, type=int)
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append', default=[])
    parser.add_argument('--mae_model_size', default='large', choices=['large', 'base'])
    parser.add_argument('--pretrained_ckpt', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\checkpoint_ViT-L_pretrain_fmow_sentinel.pth', help='MAE pretrained checkpoint path')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    # Distributed training args (aligned with main_finetune.py)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Distributed evaluation across processes')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log_dir', default='./segmentation_logs_e1', type=str)
    parser.add_argument('--output_dir', default='./segmentation_logs_e1', type=str)
    parser.add_argument('--freeze_decoder', action='store_true', default=False, help='Freeze decoder parameters')

    return parser

def build_datasets(args):
    # 8:2 切分的 Dataset（你已有）
    ds_train = SentinelIndividualImageDataset(args.julaug_dir, args.mask_dir, is_train=True, input_size=args.input_size)
    ds_val = SentinelIndividualImageDataset(args.julaug_dir, args.mask_dir, is_train=False, input_size=args.input_size)
    return ds_train, ds_val

def main(args):
    # 初始化分布式训练环境
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # 为分布式训练明确设置本地 GPU 设备
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA available"
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device)
    # 固定随机种子（包含全局rank偏移）
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    cudnn.benchmark = True

    ds_train, ds_val = build_datasets(args)

    # 采样器配置（分布式/非分布式）
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(ds_val) % num_tasks != 0:
                print('Warning: Distributed eval with non-divisible val set; duplicates may be added.')
            sampler_val = torch.utils.data.DistributedSampler(ds_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(ds_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
        sampler_val = torch.utils.data.SequentialSampler(ds_val)

    # 仅主进程创建日志目录/Writer
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dl_train = torch.utils.data.DataLoader(
        ds_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # 默认分组
    grouped_bands = args.grouped_bands if len(args.grouped_bands) > 0 else [[0,1,2,6],[3,4,5,7],[8,9]]

    # 构建 E1 模型并加载 MAE 预训练权重（Encoder+Decoder）
    model = MAEGroupChannelsSegE1(
        img_size=args.input_size, patch_size=args.patch_size, in_chans=args.in_chans,
        nb_classes=args.nb_classes, grouped_bands=tuple(tuple(b) for b in grouped_bands),
        mae_model_size=args.mae_model_size
    )
    model.load_from_mae_checkpoint(args.pretrained_ckpt)
    # freeze Decoder 或 Unfreeze Decoder
    if args.freeze_decoder:
        model.set_freeze_policy_freezedecoder()
    else:
        model.set_freeze_policy_unfreezedecoder()

    model.to(device)

    # DDP 包裹
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # 仅优化可训练参数
    optim_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    if misc.is_main_process():
        print(f"[E1] Optimizing {sum(p.numel() for p in optim_params):,} parameters")
        print(f"[E1] Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.epochs):
        # 分布式 sampler 需设置当前 epoch 以确保一致的shuffle
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, dl_train, optimizer, device, epoch, log_writer, num_classes=args.nb_classes)

        # 仅主进程执行评估与日志记录，避免多进程重复评估/写文件
        if misc.is_main_process():
            val_stats = evaluate(model, dl_val, device, num_classes=args.nb_classes)
            if log_writer is not None:
                log_writer.add_scalar('eval/OA', val_stats['OA'], epoch)
                log_writer.add_scalar('eval/mIoU', val_stats['mIoU'], epoch)
                log_writer.add_scalar('eval/mF1', val_stats['mF1'], epoch)

            # 保存日志
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "log_e1.txt"), mode="a", encoding="utf-8") as f:
                    record = {
                        'epoch': epoch,
                        'train_loss': train_stats['loss'],
                        **val_stats,
                    }
                    f.write(json.dumps(record) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print('[E1] Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
=======
import argparse
import os
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.datasets_finetune_segmentation import SentinelIndividualImageDataset
from engine_finetune_seg_e1 import train_one_epoch, evaluate
from models_mae_seg_e1 import MAEGroupChannelsSegE1

def get_args_parser():
    parser = argparse.ArgumentParser('E1: Segmentation with MAE encoder+decoder (tune decoder key parts)', add_help=False)
    parser.add_argument('--julaug_dir', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\bole_data\JulAug_croped', help='JulAug 10-band imagery root')
    parser.add_argument('--mask_dir', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\bole_data\label_post_croped', help='Segmentation masks root')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--in_chans', default=10, type=int)
    parser.add_argument('--nb_classes', default=3, type=int)
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append', default=[])
    parser.add_argument('--mae_model_size', default='large', choices=['large', 'base'])
    parser.add_argument('--pretrained_ckpt', type=str, required=False, default=r'D:\Cuselin\Project\satmae_pp-main\checkpoint_ViT-L_pretrain_fmow_sentinel.pth', help='MAE pretrained checkpoint path')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    # Distributed training args (aligned with main_finetune.py)
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Distributed evaluation across processes')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log_dir', default='./segmentation_logs_e1', type=str)
    parser.add_argument('--output_dir', default='./segmentation_logs_e1', type=str)
    return parser

def build_datasets(args):
    # 8:2 切分的 Dataset（你已有）
    ds_train = SentinelIndividualImageDataset(args.julaug_dir, args.mask_dir, is_train=True, input_size=args.input_size)
    ds_val = SentinelIndividualImageDataset(args.julaug_dir, args.mask_dir, is_train=False, input_size=args.input_size)
    return ds_train, ds_val

def main(args):
    # 初始化分布式训练环境
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    # 为分布式训练明确设置本地 GPU 设备
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA available"
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device)
    # 固定随机种子（包含全局rank偏移）
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    cudnn.benchmark = True

    ds_train, ds_val = build_datasets(args)

    # 采样器配置（分布式/非分布式）
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            ds_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(ds_val) % num_tasks != 0:
                print('Warning: Distributed eval with non-divisible val set; duplicates may be added.')
            sampler_val = torch.utils.data.DistributedSampler(ds_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(ds_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
        sampler_val = torch.utils.data.SequentialSampler(ds_val)

    # 仅主进程创建日志目录/Writer
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dl_train = torch.utils.data.DataLoader(
        ds_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # 默认分组
    grouped_bands = args.grouped_bands if len(args.grouped_bands) > 0 else [[0,1,2,6],[3,4,5,7],[8,9]]

    # 构建 E1 模型并加载 MAE 预训练权重（Encoder+Decoder）
    model = MAEGroupChannelsSegE1(
        img_size=args.input_size, patch_size=args.patch_size, in_chans=args.in_chans,
        nb_classes=args.nb_classes, grouped_bands=tuple(tuple(b) for b in grouped_bands),
        mae_model_size=args.mae_model_size
    )
    model.load_from_mae_checkpoint(args.pretrained_ckpt)
    model.set_freeze_policy()
    model.to(device)

    # DDP 包裹
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # 仅优化可训练参数
    optim_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    if misc.is_main_process():
        print(f"[E1] Optimizing {sum(p.numel() for p in optim_params):,} parameters")
        print(f"[E1] Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.epochs):
        # 分布式 sampler 需设置当前 epoch 以确保一致的shuffle
        if args.distributed:
            dl_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, dl_train, optimizer, device, epoch, log_writer, num_classes=args.nb_classes)

        # 仅主进程执行评估与日志记录，避免多进程重复评估/写文件
        if misc.is_main_process():
            val_stats = evaluate(model, dl_val, device, num_classes=args.nb_classes)
            if log_writer is not None:
                log_writer.add_scalar('eval/OA', val_stats['OA'], epoch)
                log_writer.add_scalar('eval/mIoU', val_stats['mIoU'], epoch)
                log_writer.add_scalar('eval/mF1', val_stats['mF1'], epoch)

            # 保存日志
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "log_e1.txt"), mode="a", encoding="utf-8") as f:
                    record = {
                        'epoch': epoch,
                        'train_loss': train_stats['loss'],
                        **val_stats,
                    }
                    f.write(json.dumps(record) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print('[E1] Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
>>>>>>> Stashed changes
    main(args)
