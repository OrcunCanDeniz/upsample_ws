import mmcv
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from sample_nuscenes_dataset import NuScenesPointCloudToRangeImage
from tqdm import tqdm
from skimage.measure import block_reduce
import pdb

def create_depth_target(rv, valid_mask):
    ds_factor_h = rv.shape[0] // 2
    ds_factor_w = rv.shape[1] // 64
    assert rv.shape == valid_mask.shape
    rv = rv.astype(np.float32)
    rv[~valid_mask] = np.nan
    rv_norm = rv / 55 # 0,1 -> 0,max_range
    latent_depth_mean = block_reduce(rv_norm, (ds_factor_h, ds_factor_w), func=np.nanmean)
    isnan = np.isnan(latent_depth_mean)
    latent_depth_mean[isnan] = 0
    latent_depth_std = block_reduce(rv_norm, (ds_factor_h, ds_factor_w), func=np.nanstd)
    range_head_target = np.stack([latent_depth_mean, latent_depth_std])
    return range_head_target

def generate_info(nusc, scenes, split_name, data_root, max_cam_sweeps=1, max_lidar_sweeps=1):
    # Create the SPLIT_rv directory in data root
    split_rv_dir = os.path.join(data_root, f'{split_name}_rv')
    split_head_target_dir = os.path.join(data_root, f'{split_name}_head_target')
    os.makedirs(split_rv_dir, exist_ok=True)
    os.makedirs(split_head_target_dir, exist_ok=True)
    print(f"Created directory: {split_rv_dir}")
    
    converter = NuScenesPointCloudToRangeImage( min_depth=0.0,
                                                max_depth=55.0 )
    lidar_name = 'LIDAR_TOP'
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            sweep_cam_info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info

            lidar_data = nusc.get('sample_data',
                                    cur_sample['data'][lidar_name])
            # pdb.set_trace()
            lidar_pts = np.fromfile(f"{nusc.dataroot}/{lidar_data['filename']}", dtype=np.float32).reshape(-1, 5)
            # Convert to range image using the new converter
            rv_out_name = lidar_data['filename'].split('.')[0] + '_RV.npy'
            rv_out_path = os.path.join(split_rv_dir, rv_out_name.split('/')[-1])
            range_intensity_map = converter(lidar_pts)
            np.save(rv_out_path, range_intensity_map.astype(np.float32))
            
            range_head_target = create_depth_target(range_intensity_map[...,0], range_intensity_map[..., -1].astype(np.bool))
            head_target_out_path = os.path.join(split_head_target_dir, rv_out_name.split('/')[-1])
            np.save(head_target_out_path, range_head_target.astype(np.float32))
            
            lidar_datas.append(lidar_data)
            sweep_lidar_info = dict()
            sweep_lidar_info['sample_token'] = lidar_data['sample_token']
            sweep_lidar_info['ego_pose'] = nusc.get(
                'ego_pose', lidar_data['ego_pose_token'])
            sweep_lidar_info['timestamp'] = lidar_data['timestamp']
            sweep_lidar_info['filename'] = lidar_data['filename']
            sweep_lidar_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', lidar_data['calibrated_sensor_token'])
            sweep_lidar_info['rv_path'] = "/".join(rv_out_path.split('/')[-2:]) # SPLIT_rv/RV.npy
            sweep_lidar_info['range_head_target_path'] = "/".join(head_target_out_path.split('/')[-2:]) # SPLIT_head_target/head_target.npy

            info['cam_infos'] = cam_infos
            info['lidar_info'] = sweep_lidar_info
            # for i in range(max_cam_sweeps):
            #     cam_sweeps.append(dict())


            info['cam_sweeps'] = list()
            info['lidar_sweeps'] = list()
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    data_root = './data/nuscenes/'
    trainval_nusc = NuScenes(version='v1.0-trainval',
                             dataroot=data_root,
                             verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val
    train_infos = generate_info(trainval_nusc, train_scenes, 'train', data_root)
    val_infos = generate_info(trainval_nusc, val_scenes, 'val', data_root)
    mmcv.dump(train_infos, os.path.join(data_root, 'nuscenes_upsample_infos_train.pkl'))
    mmcv.dump(val_infos, os.path.join(data_root, 'nuscenes_upsample_infos_val.pkl'))
    test_nusc = NuScenes(version='v1.0-test',
                         dataroot=data_root,
                         verbose=True)
    test_scenes = splits.test
    test_infos = generate_info(test_nusc, test_scenes, 'test', data_root)
    mmcv.dump(test_infos, os.path.join(data_root, 'nuscenes_upsample_infos_test.pkl'))


if __name__ == '__main__':
    main()
