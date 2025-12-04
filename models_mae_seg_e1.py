import torch
import torch.nn as nn
import torch.nn.functional as F

from models_mae_group_channels import mae_vit_large_patch16_dec512d8b, mae_vit_base_patch16_dec512d8b
from models_mae_group_channels import MaskedAutoencoderGroupChannelViT

import os


class MAEGroupChannelsSegE1(nn.Module):
    """
    E1: 复用预训练 Encoder + Decoder，仅微调 Decoder 的关键部分；在 MAE 重建特征上做分割适配。
    - 加载 ckpt：加载 Encoder+Decoder 全部权重（来自 MAE 预训练）。
    - 冻结策略：冻结 Encoder 和 Decoder 的非关键模块，仅解冻 decoder_pred + 上采样卷积结构。
    - 分割适配：融合 MAE 的重建图像与多尺度上采样输出，映射到 nb_classes。
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=10,
                 nb_classes=3,
                 grouped_bands=((0,1,2,6), (3,4,5,7), (8,9)),
                 mae_model_size='large'):
        super().__init__()
        self.in_chans = in_chans
        self.nb_classes = nb_classes

        if mae_model_size == 'large':
            self.mae: MaskedAutoencoderGroupChannelViT = mae_vit_large_patch16_dec512d8b(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, channel_groups=grouped_bands)
        elif mae_model_size == 'base':
            self.mae: MaskedAutoencoderGroupChannelViT = mae_vit_base_patch16_dec512d8b(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, channel_groups=grouped_bands)
        else:
            raise ValueError(f"Unsupported mae_model_size={mae_model_size}")

        # 分割适配头：融合重建图像 + 多尺度 2x/4x 上采样输出（下采到原分辨率）
        # 输入通道 = in_chans * 3
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_chans, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, nb_classes, kernel_size=1)
        )

    def load_from_mae_checkpoint(self, ckpt_path: str):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[E1] WARNING: pretrained_ckpt not found: {ckpt_path}. Skip loading and train from scratch.")
            return

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"[E1] ERROR: failed to load checkpoint {ckpt_path}: {e}")
            return

        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

        # 过滤掉明显不匹配形状的键，避免静默失败
        mae_sd = self.mae.state_dict()
        filtered = {}
        removed = []
        for k, v in state_dict.items():
            if k in mae_sd and hasattr(v, 'shape') and v.shape == mae_sd[k].shape:
                filtered[k] = v
            else:
                removed.append(k)

        if removed:
            print(f"[E1] Dropped {len(removed)} keys due to name/shape mismatch (e.g., {removed[:5]})")

        msg = self.mae.load_state_dict(filtered, strict=False)
        print(f"[E1] Loaded MAE encoder+decoder from {ckpt_path}, msg={msg}")
        loaded_keys = [k for k in filtered.keys() if k in mae_sd]
        print(f"[E1] Loaded keys count: {len(loaded_keys)} / {len(mae_sd)}")

    def set_freeze_policy_freezedecoder(self):
        # 默认全部冻结
        for p in self.mae.parameters():
            p.requires_grad = False

        # 解冻 decoder 的关键部分：decoder_pred + 上采样卷积结构（proj_up_* + up_block1, up_block2）
        for m in [
            self.mae.proj_up_conv,
            self.mae.proj_up_norm,
            self.mae.up_block1,
            self.mae.up_block2,
        ]:
            for p in m.parameters():
                p.requires_grad = True

        for head in self.mae.decoder_pred:
            for p in head.parameters():
                p.requires_grad = True

        # 分割适配头需要训练
        for p in self.seg_head.parameters():
            p.requires_grad = True

        # 显式保持以下模块冻结
        modules_to_freeze = [
            self.mae.patch_embed, self.mae.cls_token, self.mae.pos_embed,
            self.mae.channel_embed, self.mae.blocks, self.mae.norm,
            self.mae.decoder_embed, self.mae.mask_token, self.mae.decoder_pos_embed,
            self.mae.decoder_channel_embed, self.mae.decoder_blocks, self.mae.decoder_norm,
            # self.mae.proj_up_conv, self.mae.proj_up_norm, self.mae.up_block1, self.mae.up_block2,
        ]
        for m in modules_to_freeze:
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

        # 打印参数计数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"[E1] Params total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
        print("[E1] Trainable modules: decoder_pred + proj_up_conv/norm + up_block1/2 + seg_head")

    def set_freeze_policy_unfreezedecoder(self):
        # 默认全部冻结
        for p in self.mae.parameters():
            p.requires_grad = False

        # 解冻 decoder 的关键部分：decoder_pred + 上采样卷积结构（proj_up_* + up_block1, up_block2）
        for m in [
            self.mae.proj_up_conv,
            self.mae.proj_up_norm,
            self.mae.up_block1,
            self.mae.up_block2,
            self.mae.decoder_embed, self.mae.decoder_pos_embed,
            self.mae.decoder_channel_embed, self.mae.decoder_blocks, self.mae.decoder_norm,
        ]:
            for p in m.parameters():
                p.requires_grad = True

        for head in self.mae.decoder_pred:
            for p in head.parameters():
                p.requires_grad = True

        # 分割适配头需要训练
        for p in self.seg_head.parameters():
            p.requires_grad = True

        # 显式保持以下模块冻结
        modules_to_freeze = [
            self.mae.patch_embed, self.mae.cls_token, self.mae.pos_embed,
            self.mae.channel_embed, self.mae.blocks, self.mae.norm, self.mae.mask_token,
            # self.mae.decoder_embed, self.mae.decoder_pos_embed,
            # self.mae.decoder_channel_embed, self.mae.decoder_blocks, self.mae.decoder_norm,
            # self.mae.proj_up_conv, self.mae.proj_up_norm, self.mae.up_block1, self.mae.up_block2,
        ]
        for m in modules_to_freeze:
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

        # 打印参数计数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"[E1] Params total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
        print("[E1] Trainable modules: decoder_pred + proj_up_conv/norm + up_block1/2 + seg_head")

    def forward(self, imgs):
        """
        imgs: [N, in_chans, H, W]
        输出：logits [N, nb_classes, H, W]
        - 关闭掩码：mask_ratio=0.0，充分利用预训练的重建路径
        - 融合重建与多尺度上采样特征
        """
        latent, mask, ids_restore = self.mae.forward_encoder(imgs, mask_ratio=0.0)   # 不掩码
        pred_patch = self.mae.forward_decoder(latent, ids_restore)                   # [N, C, L, p*p]

        # 重建原分辨率图像
        recon = self.mae.unpatchify(
            pred_patch.permute(0, 2, 1, 3).reshape(pred_patch.shape[0], -1, self.in_chans, self.mae.patch_size**2)
            .permute(0,1,2,3).reshape(pred_patch.shape[0], pred_patch.shape[2], self.in_chans*self.mae.patch_size**2),
            self.mae.patch_embed[0].patch_size[0], self.in_chans
        )
        # 使用模型自带的多尺度上采样路径
        # pred_2x, pred_4x = self.mae.forward_multiscale(pred_patch)  # [N, in_c, 2H, 2W], [N, in_c, 4H, 4W]

        # 统一到 HxW
        # H, W = recon.shape[-2:]
        # f2 = F.interpolate(pred_2x, size=(H, W), mode='bilinear', align_corners=False)
        # f4 = F.interpolate(pred_4x, size=(H, W), mode='bilinear', align_corners=False)

        # feat = torch.cat([recon, f2, f4], dim=1)  # [N, in_c*3, H, W]
        # print("recon.shape:{}".format(recon.shape))
        logits = self.seg_head(recon)
        # print("logits.shape:{}".format(logits.shape))
=======
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_mae_group_channels import mae_vit_large_patch16_dec512d8b, mae_vit_base_patch16_dec512d8b
from models_mae_group_channels import MaskedAutoencoderGroupChannelViT

import os


class MAEGroupChannelsSegE1(nn.Module):
    """
    E1: 复用预训练 Encoder + Decoder，仅微调 Decoder 的关键部分；在 MAE 重建特征上做分割适配。
    - 加载 ckpt：加载 Encoder+Decoder 全部权重（来自 MAE 预训练）。
    - 冻结策略：冻结 Encoder 和 Decoder 的非关键模块，仅解冻 decoder_pred + 上采样卷积结构。
    - 分割适配：融合 MAE 的重建图像与多尺度上采样输出，映射到 nb_classes。
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=10,
                 nb_classes=3,
                 grouped_bands=((0,1,2,6), (3,4,5,7), (8,9)),
                 mae_model_size='large'):
        super().__init__()
        self.in_chans = in_chans
        self.nb_classes = nb_classes

        if mae_model_size == 'large':
            self.mae: MaskedAutoencoderGroupChannelViT = mae_vit_large_patch16_dec512d8b(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, channel_groups=grouped_bands)
        elif mae_model_size == 'base':
            self.mae: MaskedAutoencoderGroupChannelViT = mae_vit_base_patch16_dec512d8b(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, channel_groups=grouped_bands)
        else:
            raise ValueError(f"Unsupported mae_model_size={mae_model_size}")

        # 分割适配头：融合重建图像 + 多尺度 2x/4x 上采样输出（下采到原分辨率）
        # 输入通道 = in_chans * 3
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_chans, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, nb_classes, kernel_size=1)
        )

    def load_from_mae_checkpoint(self, ckpt_path: str):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[E1] WARNING: pretrained_ckpt not found: {ckpt_path}. Skip loading and train from scratch.")
            return

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"[E1] ERROR: failed to load checkpoint {ckpt_path}: {e}")
            return

        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

        # 过滤掉明显不匹配形状的键，避免静默失败
        mae_sd = self.mae.state_dict()
        filtered = {}
        removed = []
        for k, v in state_dict.items():
            if k in mae_sd and hasattr(v, 'shape') and v.shape == mae_sd[k].shape:
                filtered[k] = v
            else:
                removed.append(k)

        if removed:
            print(f"[E1] Dropped {len(removed)} keys due to name/shape mismatch (e.g., {removed[:5]})")

        msg = self.mae.load_state_dict(filtered, strict=False)
        print(f"[E1] Loaded MAE encoder+decoder from {ckpt_path}, msg={msg}")
        loaded_keys = [k for k in filtered.keys() if k in mae_sd]
        print(f"[E1] Loaded keys count: {len(loaded_keys)} / {len(mae_sd)}")

    def set_freeze_policy(self):
        # 默认全部冻结
        for p in self.mae.parameters():
            p.requires_grad = False

        # 解冻 decoder 的关键部分：decoder_pred + 上采样卷积结构（proj_up_* + up_block1, up_block2）
        for m in [
            self.mae.proj_up_conv,
            self.mae.proj_up_norm,
            self.mae.up_block1,
            self.mae.up_block2,
        ]:
            for p in m.parameters():
                p.requires_grad = True

        for head in self.mae.decoder_pred:
            for p in head.parameters():
                p.requires_grad = True

        # 分割适配头需要训练
        for p in self.seg_head.parameters():
            p.requires_grad = True

        # 显式保持以下模块冻结
        modules_to_freeze = [
            self.mae.patch_embed, self.mae.cls_token, self.mae.pos_embed,
            self.mae.channel_embed, self.mae.blocks, self.mae.norm,
            self.mae.decoder_embed, self.mae.mask_token, self.mae.decoder_pos_embed,
            self.mae.decoder_channel_embed, self.mae.decoder_blocks, self.mae.decoder_norm,
            # self.mae.proj_up_conv, self.mae.proj_up_norm, self.mae.up_block1, self.mae.up_block2,
        ]
        for m in modules_to_freeze:
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

        # 打印参数计数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"[E1] Params total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
        print("[E1] Trainable modules: decoder_pred + proj_up_conv/norm + up_block1/2 + seg_head")

    def forward(self, imgs):
        """
        imgs: [N, in_chans, H, W]
        输出：logits [N, nb_classes, H, W]
        - 关闭掩码：mask_ratio=0.0，充分利用预训练的重建路径
        - 融合重建与多尺度上采样特征
        """
        latent, mask, ids_restore = self.mae.forward_encoder(imgs, mask_ratio=0.0)   # 不掩码
        pred_patch = self.mae.forward_decoder(latent, ids_restore)                   # [N, C, L, p*p]

        # 重建原分辨率图像
        recon = self.mae.unpatchify(
            pred_patch.permute(0, 2, 1, 3).reshape(pred_patch.shape[0], -1, self.in_chans, self.mae.patch_size**2)
            .permute(0,1,2,3).reshape(pred_patch.shape[0], pred_patch.shape[2], self.in_chans*self.mae.patch_size**2),
            self.mae.patch_embed[0].patch_size[0], self.in_chans
        )
        # 使用模型自带的多尺度上采样路径
        # pred_2x, pred_4x = self.mae.forward_multiscale(pred_patch)  # [N, in_c, 2H, 2W], [N, in_c, 4H, 4W]

        # 统一到 HxW
        # H, W = recon.shape[-2:]
        # f2 = F.interpolate(pred_2x, size=(H, W), mode='bilinear', align_corners=False)
        # f4 = F.interpolate(pred_4x, size=(H, W), mode='bilinear', align_corners=False)

        # feat = torch.cat([recon, f2, f4], dim=1)  # [N, in_c*3, H, W]
        # print("recon.shape:{}".format(recon.shape))
        logits = self.seg_head(recon)
        # print("logits.shape:{}".format(logits.shape))
>>>>>>> Stashed changes
        return logits
