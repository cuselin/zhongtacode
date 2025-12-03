import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class GroupChannelsSegVisionTransformer(VisionTransformer):
    """
    GroupChannels ViT backbone adapted for segmentation:
      - Supports channel grouping with separate patch embeddings per group
      - Uses channel embeddings + positional embeddings
      - Exposes patch tokens (without cls) as a 2D feature map [B,C,H/ps,W/ps]
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=10, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 channel_embed=256, channel_groups=None):
        
        # 如果没有指定通道分组，为10通道Sentinel-2数据创建默认分组
        if channel_groups is None:
            if in_chans == 10:
                # Sentinel-2 10通道的合理分组：可见光+近红外、红边、短波红外
                channel_groups = ((0, 1, 2, 7), (3, 4, 5, 6), (8, 9))  # RGB+NIR, RedEdge, SWIR
            # elif in_chans == 20:
            #     # 20通道（JulAug+May拼接）的分组
            #     channel_groups = ((0, 1, 2, 7, 10, 11, 12, 17), (3, 4, 5, 6, 13, 14, 15, 16), (8, 9, 18, 19))
            else:
                # 其他情况的简单分组
                channels_per_group = max(1, in_chans // 3)
                channel_groups = tuple(
                    tuple(range(i * channels_per_group, min((i + 1) * channels_per_group, in_chans)))
                    for i in range(3)
                )
        
        self.channel_groups = channel_groups
        
        # 验证通道分组覆盖所有输入通道
        all_channels = set()
        for group in channel_groups:
            all_channels.update(group)
        assert len(all_channels) == in_chans, f"Channel groups must cover all {in_chans} channels, got {sorted(all_channels)}"
        
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)

        # 为每个通道组创建独立的patch embedding
        self.patch_embed = nn.ModuleList([
            PatchEmbed(img_size, patch_size, len(group), embed_dim)
            for group in channel_groups
        ])
        num_patches = self.patch_embed[0].num_patches

        # 位置嵌入（不包含通道维度）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - channel_embed))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 通道嵌入
        num_groups = len(channel_groups)
        self.channel_embed = nn.Parameter(torch.zeros(1, num_groups, channel_embed))
        chan_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], torch.arange(num_groups).numpy())
        self.channel_embed.data.copy_(torch.from_numpy(chan_embed).float().unsqueeze(0))

        # CLS token的通道嵌入
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, channel_embed))
        channel_cls_embed = torch.zeros((1, channel_embed))
        self.channel_cls_embed.data.copy_(channel_cls_embed.float().unsqueeze(0))

    def forward_feature_map(self, x):
        """
        Forward pass for segmentation, returns feature map without cls token
        """
        b, c, h, w = x.shape

        # 对每个通道组进行patch embedding
        x_c_embed = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, group, :, :]  # 选择当前组的通道
            x_c_embed.append(self.patch_embed[i](x_c))  # [B, N, D]

        x = torch.stack(x_c_embed, dim=1)  # [B, G, N, D]
        _, G, N, D = x.shape

        # 添加通道嵌入和位置嵌入
        channel_embed = self.channel_embed.unsqueeze(2)  # [1, G, 1, channel_embed_dim]
        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # [1, 1, N, pos_embed_dim]

        # 扩展维度以匹配
        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # [1, G, N, channel_embed_dim]
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # [1, G, N, pos_embed_dim]
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # [1, G, N, D]

        # 添加位置和通道嵌入
        x = x + pos_channel  # [B, G, N, D]
        x = x.view(b, -1, D)  # [B, G*N, D]

        # 添加CLS token
        cls_pos_channel = torch.cat((self.pos_embed[:, :1, :], self.channel_cls_embed), dim=-1)  # [1, 1, D]
        cls_tokens = cls_pos_channel + self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + G*N, D]
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # [B, 1 + G*N, D]
        patch_tokens = x[:, 1:, :]  # 去掉CLS token -> [B, G*N, D]
        
        # 重塑为 [B, G, N, D] 并在组维度上聚合为 [B, N, D]
        patch_tokens = patch_tokens.view(b, G, N, D).mean(dim=1)
        
        # 重塑为2D特征图
        H = W = int(N ** 0.5)  # N = H/patch_size * W/patch_size
        feat = patch_tokens.transpose(1, 2).reshape(b, self.embed_dim, H, W)  # [B, D, H, W]
        return feat


class ViTSegHead(nn.Module):
    """
    Simple upsampling decoder:
      - Conv refinement
      - Bilinear upsample to input size
      - 1x1 conv to num_classes
    """
    def __init__(self, in_channels: int, num_classes: int, upsample_scale: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.classifier = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        self.upsample_scale = upsample_scale

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        x = self.classifier(x)
        return x


class ViTSegModel(nn.Module):
    """
    GroupChannels ViT-Large backbone with a lightweight segmentation head.
    Supports channel grouping for multi-spectral satellite imagery.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=10, num_classes=3, 
                 channel_groups=None, channel_embed=256,
                 embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.backbone = GroupChannelsSegVisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            channel_groups=channel_groups, channel_embed=channel_embed
        )
        # Feature map size = img_size/patch_size; we upsample by patch_size
        self.head = ViTSegHead(in_channels=embed_dim, num_classes=num_classes, upsample_scale=patch_size)

    def forward(self, x):
        feat = self.backbone.forward_feature_map(x)  # [B,1024,H/ps,W/ps]
        logits = self.head(feat)                     # [B,num_classes,H,W]
        return logits

    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.channel_embed', 'backbone.channel_cls_embed', 'backbone.cls_token'}


def vit_large_seg_patch16(**kwargs):
    """
    ViT-Large segmentation model with GroupChannels support
    """
    model = ViTSegModel(
        embed_dim=1024, depth=24, num_heads=16,
        **kwargs
    )
    return model


def vit_base_seg_patch16(**kwargs):
    """
    ViT-Base segmentation model with GroupChannels support
    """
    model = ViTSegModel(
        embed_dim=768, depth=12, num_heads=12,
        **kwargs
    )
    return model