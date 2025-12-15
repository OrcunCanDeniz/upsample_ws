# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
import torch

import torch.utils.data as data
import torch.nn.functional as F
from pyquaternion import Quaternion
import mmcv
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from PIL import Image

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import ImageDataset
import numpy as np

import os
import os.path
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
import pdb
from pathlib import Path
import quaternion  # pip install numpy-quaternion

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')
NPY_EXTENSIONS = ('.npy', '.rimg', '.bin')
dataset_list = {}

def register_dataset(name):
    def decorator(cls):
        dataset_list[name] = cls
        return cls
    return decorator


def generate_dataset(args, is_train):
    dataset = dataset_list[args.dataset_select]
    return dataset(is_train, args)


def tf_mat_sensor_to_ego(cal):
    """
    Build 4x4 homogeneous transform so that (row-vector) points_at_ego = points_at_lidar @ T

    cal: dict with keys 'translation' (x,y,z) and 'rotation' (w,x,y,z) from nuScenes calibrated_sensor.
    Returns: (4,4) NumPy array T.
    
    Example:
        # Apply to lidar points (row-vector convention)
        pts_lidar = np.array([[1.0, 2.0, 3.0]])
        pts_h = np.hstack([pts_lidar, np.ones((pts_lidar.shape[0], 1))])
        pts_ego = pts_h @ T
    """
    t = np.asarray(cal['translation'], dtype=float)   # [tx, ty, tz]
    w, x, y, z = map(float, cal['rotation'])          # nuScenes uses [w, x, y, z]

    # create quaternion and normalize
    q = np.quaternion(w, x, y, z).normalized()
    R = quaternion.as_rotation_matrix(q)  # shape (3,3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mu, sigma):
        super().__init__()#
        self.sigma = sigma
        self.mu = mu
    def __call__(self, img):
        return torch.randn(img.size()) * self.sigma + self.mu


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    

class LogTransform(object):
    def __call__(self, tensor):
        return torch.log1p(tensor)


class CropRanges(object):
    def __init__(self, min_dist, max_dist):
        self.max_dist = max_dist
        self.min_dist = min_dist
    def __call__(self, tensor):
        mask = (tensor >= self.min_dist) & (tensor < self.max_dist)
        num_pixels = mask.sum()
        return torch.where(mask , tensor, 0), num_pixels

class KeepCloseScan(object):
    def __init__(self, max_dist):
        self.max_dist = max_dist
    def __call__(self, tensor):
        return torch.where(tensor < self.max_dist, tensor, 0)
    
class KeepFarScan(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist
    def __call__(self, tensor):
        return torch.where(tensor > self.min_dist, tensor, 0)
    

class RandomRollRangeMap(object):
    """Roll Range Map along horizontal direction, 
    this requires the input and output have the same width 
    (downsampled only in vertical direction)"""
    def __init__(self, h_img = 2048, shift = None):
        if shift is not None:
            self.shift = shift
        else:
            self.shift = np.random.randint(0, h_img)
    def __call__(self, tensor):
        # Assume the dimension is B C H W
        return torch.roll(tensor, shifts = self.shift, dims = -1)

class DepthwiseConcatenation(object):
    """Concatenate the image depth wise -> one channel to multi-channels input"""
    
    def __init__(self, h_high_res: int, downsample_factor: int):
        self.low_res_indices = [range(i, h_high_res+i, downsample_factor) for i in range(downsample_factor)]

    def __call__(self, tensor):
        return torch.cat([tensor[:, self.low_res_indices[i], :] for i in range(len(self.low_res_indices))], dim = 0)

class DownsampleTensor(object):
    def __init__(self, h_high_res: int, downsample_factor: int, random = False):
        if random:
            index = np.random.randint(0, downsample_factor)
        else:
            index = 0
        self.low_res_index = range(0+index, h_high_res+index, downsample_factor)
    def __call__(self, tensor):
        return tensor[:, self.low_res_index, :]
    
class DownsampleTensorWidth(object):
    def __init__(self, w_high_res: int, downsample_factor: int, random = False):
        if random:
            index = np.random.randint(0, downsample_factor)
        else:
            index = 0
        self.low_res_index = range(0+index, w_high_res+index, downsample_factor)
    def __call__(self, tensor):
        return tensor[:, :, self.low_res_index]

class ScaleTensor(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, tensor):
        return tensor*self.scale_factor
    
class FilterInvalidPixels(object):
    ''''Filter out pixels that are out of lidar range'''
    def __init__(self, min_range, max_range = 1):
        self.max_range = max_range
        self.min_range = min_range

    def __call__(self, tensor):
        return torch.where((tensor >= self.min_range) & (tensor <= self.max_range), tensor, 0)
    
class FilterInvalidPixelsWMask(object):
    ''''Filter out pixels that are out of lidar range'''
    def __init__(self, min_range, max_range = 1):
        self.max_range = max_range
        self.min_range = min_range

    def __call__(self, rv, mask):
        range_mask = (rv >= self.min_range) & (rv <= self.max_range)
        mask = mask & range_mask
        return torch.where(range_mask, rv, 0), mask


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    

# def npy_loader(path: str) -> np.ndarray:
#     with open(path, "rb") as f:
#         range_map = np.load(f)
#     return range_map.astype(np.float32)

def bin_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_intensity_map = np.fromfile(f, dtype=np.float32).reshape(64, 1024, 2)
        # range_map = range_intensity_map[..., 0]
    return range_intensity_map

def npy_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_intensity_map = np.load(f)
        range_map = range_intensity_map[..., 0]
    return range_map.astype(np.float32)

def mdim_npy_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.load(f)
    return data.astype(np.float32)
    
def rimg_loader(path: str) -> np.ndarray:
    """
    Read range image from .rimg file (for CARLA dataset)
    """
    with open(path, 'rb') as f:
        size =  np.fromfile(f, dtype=np.uint, count=2)
        range_image = np.fromfile(f, dtype=np.float16)
    
    range_image = range_image.reshape(size[1], size[0])
    range_image = range_image.transpose()


    return np.flip(range_image).astype(np.float32)


class RVWithImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        high_res_transform: Optional[Callable] = None,
        low_res_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = mdim_npy_loader,
        img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                        img_std=[58.395, 57.12, 57.375],
                        to_rgb=True),
        info_file: str = None,
        final_dim = (256, 704),
    ):
        # self.class_dir = class_dir
        super().__init__()
        
        self.data_root = root
        self.info_path = info_file
        self.split = 'train' if 'train' in info_file else 'val'
        info_path = os.path.join(self.data_root, self.info_path)
        self.infos = mmcv.load(info_path)
        self.cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
        self.high_res_transform = high_res_transform
        self.low_res_transform = low_res_transform
        self.loader = loader
        
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf.get('to_rgb', True)
        self.final_dim = final_dim
        self.max_range = 55
        self.min_range = 0
        self.depth_target_size = (2,64)
        
             
    def img_transform(self, img):
        W, H = img.size
        fH, fW = self.final_dim
        
        if H == fH and W == fW:
            return img, 1.0
        
        resize_scale = max(fH / H, fW / W)
        resize_dims = (int(W * resize_scale), int(H * resize_scale))
        
        # adjust image
        img = img.resize(resize_dims)
        return img, resize_scale
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return [""], {"":0}
    
    def __len__(self):
        return len(self.infos)
    
    def get_image(self, cam_infos, cams):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        cam_infos = [cam_infos]
        assert len(cam_infos) > 0

        sweep_imgs = list()
        sweep_intrin_mats = list()
        sweep_lidar2img_rts = list()
        sweep_lidar2cam_rts = list()
        image_metas = list()

        for cam in cams:
            imgs = list()
            intrin_mats = list()
            key_info = cam_infos[0]
            lidar2img_rts = list()
            lidar2cam_rts = list()
            
            # TODO: change this if needed to add ida
            # resize, resize_dims, crop, flip, \
            #     rotate_ida = self.sample_ida_augmentation(
            #         )
            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                intrin_mat = np.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = np.array(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])

                img, scale = self.img_transform(img)
                
                lidar2cam_r = np.linalg.inv(cam_info[cam]['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    cam]['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2img_rt = (intrin_mat @ lidar2cam_rt.T)
                
                if scale != 1.0:
                    scale_factor = np.eye(4)
                    scale_factor[0, 0] *= scale
                    scale_factor[1, 1] *= scale
                    lidar2img_rt = scale_factor @ lidar2img_rt

                lidar2img_rts.append(torch.from_numpy(lidar2img_rt).float())
                lidar2cam_rts.append(torch.from_numpy(lidar2cam_rt).float())
                
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(torch.from_numpy(intrin_mat).float())
                
            sweep_imgs.append(torch.stack(imgs))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_lidar2img_rts.append(torch.stack(lidar2img_rts))
            sweep_lidar2cam_rts.append(torch.stack(lidar2cam_rts))

        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        # img_metas = dict(
        #     box_type_3d=LiDARInstance3DBoxes,
        #     ego2global_translation=ego2global_translation,
        #     ego2global_rotation=ego2global_rotation,
        # )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.cat(sweep_lidar2img_rts, dim=0),
            torch.stack(sweep_lidar2cam_rts).permute(1, 0, 2, 3),
            torch.tensor(img.shape[1:])
        ] # not returning image_metas but keep popullating it in case we need it later

        return ret_list
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample_info = self.infos[index]
        rv_path = os.path.join(self.data_root, sample_info['lidar_info']['rv_path'])
        np_data = self.loader(rv_path)
        rv_sample = np_data[..., 0]
        mask_sample = np_data[..., -1]
        range_mask = np.logical_and(rv_sample >= self.min_range, rv_sample <= self.max_range)
        mask_sample = np.logical_and(mask_sample, range_mask) # pixel should satisfy both range and mask
        # Load precomputed range_head_target from disk instead of computing on-the-fly
        # head_target_rel = sample_info['lidar_info'].get('range_head_target_path')
        # if head_target_rel is None:
        #     raise KeyError('range_head_target_path missing in lidar_info')
        # head_target_path = os.path.join(self.data_root, head_target_rel)
        # range_head_target = mdim_npy_loader(head_target_path)[0]
        low_res_rv = self.low_res_transform(rv_sample)
        high_res_rv = self.high_res_transform(rv_sample)
        
        lidar2ego_mat = tf_mat_sensor_to_ego(sample_info['lidar_info']['calibrated_sensor'])
            
        image_list = self.get_image(sample_info['cam_infos'], self.cam_names)
        (
            sweep_imgs,
            sweep_intrins,
            sweep_lidar2img_rts,
            sweep_lidar2cam_rts,
            img_shape,
        ) = image_list
        ret_list = [
            sweep_imgs,
            sweep_intrins,
            sweep_lidar2img_rts,
            sweep_lidar2cam_rts,
            low_res_rv,
            high_res_rv,
            lidar2ego_mat,
            torch.from_numpy(mask_sample),
            img_shape,
            sample_info['sample_token']
            # torch.from_numpy(range_head_target)
        ]

        return ret_list

def collate_fn(data):
    imgs_batch = list()
    intrin_mats_batch = list()
    lr_rv_samples_batch = list()
    hr_rv_samples_batch = list()
    sweep_lidar2img_rts_batch = list()
    img_shapes_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_intrins,
            sweep_lidar2img_rts,
            sweep_lidar2cam_rts,
            lr_rv_sample,
            hr_rv_sample,
            lidar2ego_mat_tmp,
            mask_sample,
            img_shape,
            sample_token,
        ) = iter_data

        
        imgs_batch.append(sweep_imgs)
        lr_rv_samples_batch.append(lr_rv_sample)
        hr_rv_samples_batch.append(hr_rv_sample)
        sweep_lidar2img_rts_batch.append(sweep_lidar2img_rts)
        img_shapes_batch.append(img_shape)
    ret_list = [
        torch.stack(imgs_batch).contiguous(),
        torch.stack(lr_rv_samples_batch).contiguous(),
        torch.stack(hr_rv_samples_batch).contiguous(),
        torch.stack(sweep_lidar2img_rts_batch).contiguous(),
        torch.stack(img_shapes_batch).contiguous(),
    ]

    return ret_list

class RangeMapFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = npy_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_dir: bool = True,
    ):
        self.class_dir = class_dir
        super().__init__(
            root,
            loader,
            NPY_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
            
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        if self.class_dir:
            return super().find_classes(directory)    
        else:
            return [""], {"":0}
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        name = os.path.basename(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return {'sample': sample,
                'class':target,
                'name': name}

@register_dataset('nuscenes')
def build_nuscenes_upsampling_dataset(is_train, args):
    input_size = (8,1024)
    output_size = (32,1024)
    
    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 0, max_range = 55/80)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 0, max_range = 55/80)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0],))
    if output_size[1] // input_size[1] > 1:
        t_low_res.append(DownsampleTensorWidth(w_high_res=output_size[1], downsample_factor=output_size[1]//input_size[1],))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)        
    
    root_low_res = os.path.join(args.data_path_low_res, 'train_rv' if is_train else 'val_rv')
    root_high_res = os.path.join(args.data_path_high_res, 'train_rv' if is_train else 'val_rv')

    dataset_low_res = RangeMapFolder(root_low_res, transform = transform_low_res, loader= npy_loader, class_dir=False)
    dataset_high_res = RangeMapFolder(root_high_res, transform = transform_high_res, loader =  npy_loader, class_dir = False)

    assert len(dataset_high_res) == len(dataset_low_res)

    dataset_concat = PairDataset(dataset_low_res, dataset_high_res)
    return dataset_concat

@register_dataset('nuscenes_with_image')
def build_nuscenes_w_image_upsampling_dataset(is_train, args):
    input_size = (8,1024)
    output_size = (32,1024)
    
    t_low_res = [transforms.ToTensor(), ScaleTensor(1/55)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/55)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0],))
    if output_size[1] // input_size[1] > 1:
        t_low_res.append(DownsampleTensorWidth(w_high_res=output_size[1], downsample_factor=output_size[1]//input_size[1],))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)        
    
    info_file = "nuscenes_upsample_infos_train.pkl" if is_train else "nuscenes_upsample_infos_val.pkl"

    tmp_dir = os.environ.get("TMPDIR")
    use_work_dir = os.environ.get("USE_WORK", "0")
    if tmp_dir is not None and use_work_dir != "1":
        nusc_root = os.path.join(tmp_dir, "nusc_dataset")
    else:
        nusc_root = "./data/nuscenes"
    print("Nuscenes root directory:", nusc_root)
    dset = RVWithImageDataset(nusc_root, high_res_transform = transform_high_res, low_res_transform = transform_low_res, info_file = info_file)

    return dset


@register_dataset('durlar')
def build_durlar_upsampling_dataset(is_train, args):
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/120), FilterInvalidPixels(min_range = 0.3/120, max_range = 1)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/120), FilterInvalidPixels(min_range = 0.3/120, max_range = 1)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0]))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())
    
    if is_train and args.roll: 
        # t_low_res.append(AddGaussianNoise(sigma=0.03, mu=0))
        roll_low_res = RandomRollRangeMap()
        roll_high_res = RandomRollRangeMap(shift = roll_low_res.shift)
        t_low_res.append(roll_low_res)
        t_high_res.append(roll_high_res)

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)

    root_low_res = os.path.join(args.data_path_low_res, 'train' if is_train else 'val')
    root_high_res = os.path.join(args.data_path_high_res, 'train' if is_train else 'val')

    dataset_low_res = RangeMapFolder(root_low_res, transform = transform_low_res, loader= npy_loader, class_dir=False)
    dataset_high_res = RangeMapFolder(root_high_res, transform = transform_high_res, loader =  npy_loader, class_dir = False)


    assert len(dataset_high_res) == len(dataset_low_res)

    dataset_concat = PairDataset(dataset_low_res, dataset_high_res)
    return dataset_concat

@register_dataset('kitti')
def build_kitti_upsampling_dataset(is_train, args):
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0],))
    if output_size[1] // input_size[1] > 1:
        t_low_res.append(DownsampleTensorWidth(w_high_res=output_size[1], downsample_factor=output_size[1]//input_size[1],))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)        

    root_low_res = os.path.join(args.data_path_low_res, 'train' if is_train else 'val')
    root_high_res = os.path.join(args.data_path_high_res, 'train' if is_train else 'val')


    dataset_low_res = RangeMapFolder(root_low_res, transform = transform_low_res, loader= npy_loader, class_dir = False)
    dataset_high_res = RangeMapFolder(root_high_res, transform = transform_high_res, loader = npy_loader, class_dir = False)

    assert len(dataset_high_res) == len(dataset_low_res)

    dataset_concat = PairDataset(dataset_low_res, dataset_high_res)
    return dataset_concat
    

@register_dataset('carla')
def build_carla_upsampling_dataset(is_train, args):
    # Carla dataset is not normalized
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)
    input_img_path = str(input_size[0]) + '_' + str(input_size[1])
    output_img_path = str(output_size[0]) + '_' + str(output_size[1])

    available_resolution = os.listdir(os.path.join(args.data_path_low_res, 'Town01'))

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 2/80, max_range = 1)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 2/80, max_range = 1)]


    INPUT_DATA_UNAVAILABLE = input_img_path not in available_resolution and output_img_path in available_resolution

    if INPUT_DATA_UNAVAILABLE:
        print("There is no data for the specified input size but output size is available, Downsample input data from the output")
        t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0], ))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)

    scene_ids = ['Town01',
                 'Town02',
                 'Town03',
                 'Town04',
                 'Town05',
                 'Town06',] if is_train else ['Town07', 'Town10HD']

    scenes_data_input = []
    scenes_data_output = []
    
    for scene_ids_i in scene_ids:
        if INPUT_DATA_UNAVAILABLE:
            input_scene_datapath = os.path.join(args.data_path_low_res, scene_ids_i, output_img_path)
            output_scene_datapath = os.path.join(args.data_path_high_res, scene_ids_i, output_img_path)
            scenes_data_input.append(RangeMapFolder(input_scene_datapath, transform = transform_low_res, loader=rimg_loader, class_dir=False))
            scenes_data_output.append(RangeMapFolder(output_scene_datapath, transform = transform_high_res, loader=rimg_loader, class_dir=False))

        else:

            input_scene_datapath = os.path.join(args.data_path_low_res, scene_ids_i, input_img_path)
            output_scene_datapath = os.path.join(args.data_path_high_res, scene_ids_i, output_img_path)
            scenes_data_input.append(RangeMapFolder(input_scene_datapath, transform = transform_low_res, loader=rimg_loader, class_dir=False))
            scenes_data_output.append(RangeMapFolder(output_scene_datapath, transform = transform_high_res, loader=rimg_loader, class_dir=False))

    
    input_data = data.ConcatDataset(scenes_data_input)
    output_data = data.ConcatDataset(scenes_data_output)

    carla_dataset = PairDataset(input_data, output_data)

    return carla_dataset

if __name__ == "__main__":
    dset = build_nuscenes_upsampling_dataset(True, None)
    data = dset[0]
    b_data = collate_fn([data])
    pdb.set_trace()