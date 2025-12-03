# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
#import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ >= "0.3.2" 
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets_finetune import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
import models_vit_group_channels


from engine_finetune import (train_one_epoch, evaluate)


def get_args_parser():
    parser = argparse.ArgumentParser('SatMAE++ fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_type', default='group_c', choices=['group_c', 'vanilla'],
                        help='Use channel model')
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=96, type=int, help='images input size')
    parser.add_argument('--patch_size', default=8, type=int, help='patch embedding patch size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1., help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='./output_dir/checkpoint-50.pth', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--base_path', default='dataset/eurosat/2750/', type=str, help='dataset folder path') # for euorsat dataset
    parser.add_argument('--train_path', default='dataset/fmow_sentinel/train.csv', type=str,
                        help='Train .csv path')
    parser.add_argument('--test_path', default='dataset/fmow_sentinel/val.csv', type=str,
                        help='Test .csv path')
    parser.add_argument('--dataset_type', default='sentinel', choices=['rgb', 'sentinel', 'euro_sat', 'resisc', 'ucmerced'],
                        help='Whether to use fmow rgb, sentinel, or other dataset.')
    parser.add_argument('--masked_bands', default=None, nargs='+', type=int,
                        help='Sequence of band indices to mask (with mean val) in sentinel dataset')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Which bands (0 indexed) to drop from sentinel data.")
    parser.add_argument('--grouped_bands', type=int, nargs='+', action='append',
                        default=[], help="Bands to group for GroupC vit")

    parser.add_argument('--nb_classes', default=62, type=int, help='number of the classification types') # 62 classes for fmow (rgb or sentinel)
    parser.add_argument('--output_dir', default='./finetune_logs', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./finetune_logs', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1, help='How frequently (in epochs) to save ckpt')
    parser.add_argument('--wandb', type=str, default=None, help="Wandb project name, eg: sentinel_finetune")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args) # 初始化分布式训练环境

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device) # 确定当前进程使用的设备（GPU或CPU）

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True  # 启用cuDNN的自动优化

    dataset_train = build_fmow_dataset(is_train=True, args=args) # 构建训练数据集
    dataset_val = build_fmow_dataset(is_train=False, args=args) # 构建验证数据集

    if True:  # args.distributed: # 检查是否启用了分布式训练
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        ) # 分布式训练采样器，每个进程随机采样一部分数据
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval: # 检查是否启用了分布式评估
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias （在分布式评估时，为了减少评估 bias，建议启用 shuffle）
        else:   # 非分布式评估，使用顺序采样器
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:   # 非分布式训练，使用随机采样器
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 检查当前进程是否是主进程（rank 0），如果是，且启用了日志记录目录，且不是评估模式，则创建日志目录
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    # 构建训练数据加载器    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # 构建验证数据加载器
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    # 构建 Mixup 函数（如果启用了 Mixup 或 Cutmix）
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active: # 检查是否启用了 Mixup 或 Cutmix
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes) # 初始化 Mixup 函数
        #
    
    # Define the model
    if args.model_type == 'group_c':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]

        print(f"Grouping bands {args.grouped_bands}")
        # 从模型字典中获取指定的模型架构
        model = models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            channel_groups=args.grouped_bands, num_classes=args.nb_classes,
            drop_path_rate=args.drop_path, global_pool=args.global_pool
        )
    else:
        model = models_vit.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size, in_chans=dataset_train.in_c,
            num_classes=args.nb_classes, drop_path_rate=args.drop_path, global_pool=args.global_pool,
        )
    
    
    if args.finetune and not args.eval: # 检查是否启用了微调（finetune），且不是评估模式（eval）
        checkpoint = torch.load(args.finetune, map_location='cpu') # 从指定路径加载预训练检查点

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] # 从检查点中提取模型状态字典
        state_dict = model.state_dict() # 获取当前模型的状态字典

        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]

        # TODO: Do something smarter?
        # 从模型字典中获取指定的模型架构，检查是否有key在pth中不存在于当前模型中
        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # interpolate position embedding （如果检查点中的位置嵌入与当前模型不同，进行插值）
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # 检查是否有缺失的键（missing_keys），如果有，打印并断言它们是否与全局池（global_pool）相关
        # TODO: change assert msg based on patch_embed
        if args.global_pool:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer 
        trunc_normal_(model.head.weight, std=2e-5) # 手动初始化分类头的权重（fc层），使用截断正态分布初始化
    

    model.to(device) ## 将模型移动到指定的设备

    model_without_ddp = model # DDP通过在多个GPU上复制模型副本，并并行处理数据来实现训练加速。
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size() # 计算有效批量大小（考虑分布式训练）

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256 # 计算实际学习率（考虑批量大小缩放）

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd) 
    # 构建优化器，考虑层间学习率衰减（layer-wise lr decay）
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler() # 用于混合精度训练的损失缩放器（loss scaler）

    # 构建损失函数（criterion），考虑混合标签平滑（mixup label smoothing）
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    # 加载预训练模型（如果指定了finetune路径）
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    '''
    if global_rank == 0 and args.wandb is not None:
        wandb.init(project=args.wandb, entity="mae-sentinel")
        wandb.config.update(args)
        wandb.watch(model)
    '''

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
              f"acc5: {test_stats['acc5']:.2f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            '''
            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Invalid stats?")
            '''

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
