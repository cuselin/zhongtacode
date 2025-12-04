import os
import pandas as pd
import numpy as np
import warnings
import random
import json
import cv2
import re
from glob import glob
from typing import Any, Optional, List, Dict, Tuple
import rasterio
from rasterio import logging

import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

#########################################################
# SENTINEL DEFINITIONS
#########################################################
class SentinelNormalize:
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [ 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,   1732.16362238, 1247.91870117]
    std = [ 650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  1310.36996126, 1087.6020813]

    def __init__(self,
                 julaug_dir: str,
                 mask_dir: str,
                 input_size: int = 224,
                 is_train: bool = True,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 ignore_index: Optional[int] = None):
        """
        仅使用 JulAug 影像的语义分割数据集：
        - 扫描掩码目录，按文件名后4位编号匹配 JulAug 影像。
        - 返回图像张量[C,H,W]与像素级标签[H,W]。
        """
        super().__init__(in_c=10)
        self.julaug_dir = julaug_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.in_c = 10

        # 基于掩码文件按后四位编号配对 JulAug 影像
        all_samples: List[Dict[str, str]] = []
        mask_files = [f for f in os.listdir(self.mask_dir) if f.lower().endswith('.tif')]
        for mf in mask_files:
            mid = self._suffix_id(mf)
            if not mid:
                continue
            jul_candidates = [os.path.join(self.julaug_dir, f) for f in os.listdir(self.julaug_dir)
                              if f.lower().endswith('.tif') and self._suffix_id(f) == mid]
            if len(jul_candidates) == 0:
                continue
            all_samples.append({'img': jul_candidates[0], 'mask': os.path.join(self.mask_dir, mf)})

        if len(all_samples) == 0:
            raise RuntimeError(f"No matched samples found in {self.julaug_dir} and {self.mask_dir} by 4-digit suffix.")

        # 训练验证集划分 (8:2)
        all_samples.sort(key=lambda x: x['img'])  # 确保划分的一致性
        total_samples = len(all_samples)
        train_size = int(0.8 * total_samples)
        
        if is_train:
            self.samples = all_samples[:train_size]
            print(f"Training set: {len(self.samples)} samples")
        else:
            self.samples = all_samples[train_size:]
            print(f"Validation set: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def open_image(self, img_path):
        # 读取多波段tif为(H,W,C)的float32数组（JulAug 10通道）
        with rasterio.open(img_path) as data:
            arr = data.read()  # (C,H,W)
        return arr.transpose(1, 2, 0).astype(np.float32)  # (H,W,C)

    def __getitem__(self, idx):
        """
        返回像素级分割样本：
        - 图像：JulAug 10通道
        - 掩码：单通道整型数组[H,W]
        - 统一变换：对图像与掩码同步resize到input_size；图像ToTensor与可选Normalize；可选随机水平翻转
        """
        item = self.samples[idx]
        img_np = self.open_image(item['img'])        # (H,W,10)
        mask_np = self._read_mask_tif(item['mask'])  # (H,W) int

        # 变换：同步resize；图像ToTensor与Normalize；掩码long
        img_t, mask_t = self._apply_joint_transform(img_np, mask_np)

        return {'img': img_t, 'label': mask_t}

    # -------------------- 辅助方法（类内部，不影响外部文件） --------------------
    @staticmethod
    def _suffix_id(filename: str) -> Optional[str]:
        # 提取文件名（不含扩展名）末尾4位数字作为样本编号
        stem = os.path.splitext(os.path.basename(filename))[0]
        m = re.search(r'(\d{4})$', stem)
        return m.group(1) if m else None

    @staticmethod
    def _read_mask_tif(path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(1)  # (H,W)
        return arr.astype(np.int64)

    def _apply_joint_transform(self, img_np: np.ndarray, mask_np: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # 若尺寸不同于input_size，图像用BICUBIC，掩码用NEAREST缩放到一致大小
        h, w = img_np.shape[:2]
        if (h, w) != (self.input_size, self.input_size):
            img_pil = Image.fromarray(img_np.astype(np.uint8)) if img_np.dtype == np.uint8 else Image.fromarray(img_np)
            img_pil = img_pil.resize((self.input_size, self.input_size), resample=Image.BICUBIC)
            mask_pil = Image.fromarray(mask_np.astype(np.int32))
            mask_pil = mask_pil.resize((self.input_size, self.input_size), resample=Image.NEAREST)
            img_np = np.array(img_pil)
            mask_np = np.array(mask_pil)

        # 图像ToTensor与可选Normalize
        img_t = transforms.ToTensor()(img_np)  # [C,H,W]
        if self.mean is not None and self.std is not None:
            img_t = transforms.Normalize(mean=self.mean, std=self.std)(img_t)

        # 掩码为LongTensor
        mask_t = torch.from_numpy(mask_np).long()

        # 简单增强：训练时随机水平翻转（同步图像与掩码）
        if self.is_train and torch.rand(1).item() < 0.5:
            img_t = torch.flip(img_t, dims=[2])   # flip width
            mask_t = torch.flip(mask_t, dims=[1])

        return img_t, mask_t

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # use specific Sentinel normalization to avoid NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)

################################################################################################################

def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Initializes a SatelliteDataset object given provided args.
    :param is_train: Whether we want the dataset for training or evaluation
    :param args: Argparser args object with provided arguments
    :return: SatelliteDataset object.
    """
    file_path = os.path.join(args.train_path if is_train else args.test_path)
    
    if args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(file_path, transform, masked_bands=args.masked_bands,
                                                 dropped_bands=args.dropped_bands)
    
    print(dataset)

    return dataset
