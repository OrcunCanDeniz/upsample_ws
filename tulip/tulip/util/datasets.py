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
        loader: Callable[[str], Any] = npy_loader,
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
            
    def img_transform(self, img):
        W, H = img.size
        fH, fW = self.final_dim
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])

        ida_mat = ida_rot.new_zeros(4, 4)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran
        return img, ida_mat
        
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
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()

        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            key_info = cam_infos[0]
            # TODO: change this if needed to add ida
            # resize, resize_dims, crop, flip, \
            #     rotate_ida = self.sample_ida_augmentation(
            #         )
            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])

                img, ida_mat = self.img_transform(img)
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))

        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]

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
        rv_sample = self.loader(rv_path)
        low_res_rv = self.low_res_transform(rv_sample)
        high_res_rv = self.high_res_transform(rv_sample)
        
        lidar2ego_mat = tf_mat_sensor_to_ego(sample_info['lidar_info']['calibrated_sensor'])
            
        image_list = self.get_image(sample_info['cam_infos'], self.cam_names)
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_list[:7]
        img_metas['token'] = sample_info['sample_token']
        bda_mat = torch.eye(4)
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            low_res_rv,
            high_res_rv,
            lidar2ego_mat,
        ]

        return ret_list

def collate_fn(data):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    img_metas_batch = list()
    lr_rv_samples_batch = list()
    hr_rv_samples_batch = list()
    lidar2ego_mat = None
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            lr_rv_sample,
            hr_rv_sample,
            lidar2ego_mat_tmp,
        ) = iter_data[:11]
        
        if lidar2ego_mat is None:
            lidar2ego_mat = lidar2ego_mat_tmp
        
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        lr_rv_samples_batch.append(lr_rv_sample)
        hr_rv_samples_batch.append(hr_rv_sample)
        
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        torch.stack(lr_rv_samples_batch),
        torch.stack(hr_rv_samples_batch),
        torch.from_numpy(lidar2ego_mat).float(),
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
def build_nuscenes_w_image_upsampling_dataset(is_train, log_transform = False):
    input_size = (8,1024)
    output_size = (32,1024)
    
    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 0, max_range = 55/80)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 0, max_range = 55/80)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0],))
    if output_size[1] // input_size[1] > 1:
        t_low_res.append(DownsampleTensorWidth(w_high_res=output_size[1], downsample_factor=output_size[1]//input_size[1],))

    if log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)        
    
    info_file = "nuscenes_upsample_infos_train.pkl" if is_train else "nuscenes_upsample_infos_val.pkl"

    nusc_root = "./data/nuscenes"
    dset = RVWithImageDataset(nusc_root, high_res_transform = transform_high_res, low_res_transform = transform_low_res, loader = npy_loader, info_file = info_file)

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