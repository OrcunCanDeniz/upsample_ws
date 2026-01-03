import numpy as np
import math
import torch
from chamfer_distance import ChamferDistance as chamfer_dist
# from pyemd import emd
import pdb
ELEV_DEG_PER_RING_NUCSENES = np.array([-30.67, -29.33, -28., -26.66, -25.33, -24., -22.67, -21.33,
                               -20., -18.67, -17.33, -16., -14.67, -13.33, -12., -10.67,
                                -9.33, -8., -6.66, -5.33, -4., -2.67, -1.33, 0.,
                                1.33, 2.67, 4., 5.33, 6.67, 8., 9.33, 10.67], dtype=np.float32)
ELEV_DEG_PER_RING_NUCSENES_RAD = np.deg2rad(ELEV_DEG_PER_RING_NUCSENES)
ELEV_DEG_PER_RING_NUCSENES_RAD_TORCH = torch.from_numpy(ELEV_DEG_PER_RING_NUCSENES_RAD).float()
offset_lut = np.array([48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0])

azimuth_lut = np.array([4.23,1.43,-1.38,-4.18,4.23,1.43,-1.38,-4.18,4.24,1.43,-1.38,-4.18,4.24,1.42,-1.38,-4.19,4.23,1.43,-1.38,-4.19,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.39,-4.2,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.4,-4.2,4.23,1.42,-1.4,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.39,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.41,-4.21,4.22,1.41,-1.41,-4.21,4.21,1.4,-1.41,-4.21,4.21,1.41,-1.41,-4.21,4.22,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.4,-1.41,-4.23,4.21,1.4,-1.42,-4.23,4.21,1.4,-1.42,-4.22,4.21,1.39,-1.42,-4.22,4.21,1.4,-1.42,-4.21,4.21,1.4,-1.42,-4.22,4.2,1.4,-1.41,-4.22,4.2,1.4,-1.42,-4.22,4.2,1.4,-1.42,-4.22])

elevation_lut = np.array([21.42,21.12,20.81,20.5,20.2,19.9,19.58,19.26,18.95,18.65,18.33,18.02,17.68,17.37,17.05,16.73,16.4,16.08,15.76,15.43,15.1,14.77,14.45,14.11,13.78,13.45,13.13,12.79,12.44,12.12,11.77,11.45,11.1,10.77,10.43,10.1,9.74,9.4,9.06,8.72,8.36,8.02,7.68,7.34,6.98,6.63,6.29,5.95,5.6,5.25,4.9,4.55,4.19,3.85,3.49,3.15,2.79,2.44,2.1,1.75,1.38,1.03,0.68,0.33,-0.03,-0.38,-0.73,-1.07,-1.45,-1.8,-2.14,-2.49,-2.85,-3.19,-3.54,-3.88,-4.26,-4.6,-4.95,-5.29,-5.66,-6.01,-6.34,-6.69,-7.05,-7.39,-7.73,-8.08,-8.44,-8.78,-9.12,-9.45,-9.82,-10.16,-10.5,-10.82,-11.19,-11.52,-11.85,-12.18,-12.54,-12.87,-13.2,-13.52,-13.88,-14.21,-14.53,-14.85,-15.2,-15.53,-15.84,-16.16,-16.5,-16.83,-17.14,-17.45,-17.8,-18.11,-18.42,-18.72,-19.06,-19.37,-19.68,-19.97,-20.31,-20.61,-20.92,-21.22])

origin_offset = 0.015806

lidar_to_sensor_z_offset = 0.03618

angle_off = math.pi * 4.2285/180.

def idx_from_px(px, cols):
    vv = (px[:,0].astype(int) + cols - offset_lut[px[:, 1].astype(int)]) % cols
    idx = px[:, 1] * cols + vv
    return idx


def px_to_xyz(px, p_range, cols): # px: (u, v) size = (H*W,2)
    u = (cols + px[:,0]) % cols
    azimuth_radians = math.pi * 2.0 / cols 
    encoder = 2.0 * math.pi - (u * azimuth_radians) 
    azimuth = angle_off
    elevation = math.pi * elevation_lut[px[:, 1].astype(int)] / 180.

    x_lidar = (p_range - origin_offset) * np.cos(encoder+azimuth)*np.cos(elevation) + origin_offset*np.cos(encoder)
    y_lidar = (p_range - origin_offset) * np.sin(encoder+azimuth)*np.cos(elevation) + origin_offset*np.sin(encoder)
    z_lidar = (p_range - origin_offset) * np.sin(elevation) 
    x_sensor = -x_lidar
    y_sensor = -y_lidar
    z_sensor = z_lidar + lidar_to_sensor_z_offset
    return np.stack((x_sensor, y_sensor, z_sensor), axis=-1)

def voxelize_point_cloud_torch(point_cloud, grid_size, min_coord, max_coord):
    """
    Voxelize a single point cloud using PyTorch.
    
    Args:
        point_cloud: torch.Tensor of shape (N, 3), N = number of points
        grid_size: float, size of each voxel
        min_coord: torch.Tensor of shape (3,), minimum coordinates
        max_coord: torch.Tensor of shape (3,), maximum coordinates
    
    Returns:
        voxel_grid: torch.BoolTensor of shape (D, H, W)
    """
    device = point_cloud.device
    
    # Calculate dimensions
    dimensions = ((max_coord - min_coord) / grid_size).long() + 1
    voxel_grid = torch.zeros(tuple(dimensions.tolist()), dtype=torch.bool, device=device)
    
    # Assign points to voxels
    indices = ((point_cloud - min_coord) / grid_size).long()
    
    # Clamp indices to valid range
    for i in range(3):
        indices[:, i] = torch.clamp(indices[:, i], 0, dimensions[i] - 1)
    
    # Remove duplicate indices
    indices_unique = torch.unique(indices, dim=0)
    
    # Set voxels to True
    voxel_grid[indices_unique[:, 0], indices_unique[:, 1], indices_unique[:, 2]] = True
    
    return voxel_grid


def calculate_metrics_torch(voxel_grid_predicted, voxel_grid_ground_truth):
    """
    Calculate IoU, precision, and recall using PyTorch for a single sample.
    
    Args:
        voxel_grid_predicted: torch.BoolTensor of shape (D, H, W)
        voxel_grid_ground_truth: torch.BoolTensor of shape (D, H, W)
    
    Returns:
        iou: scalar tensor
        precision: scalar tensor
        recall: scalar tensor
    """
    # Flatten spatial dimensions
    pred_flat = voxel_grid_predicted.flatten()
    gt_flat = voxel_grid_ground_truth.flatten()
    
    # Calculate intersection and union
    intersection = torch.logical_and(pred_flat, gt_flat)
    union = torch.logical_or(pred_flat, gt_flat)
    
    # Calculate metrics
    intersection_sum = intersection.sum().float()
    union_sum = union.sum().float()
    pred_sum = pred_flat.sum().float()
    gt_sum = gt_flat.sum().float()
    
    # IoU
    iou = intersection_sum / (union_sum + 1e-8)
    
    # Precision and Recall
    true_positive = intersection_sum
    false_positive = pred_sum - true_positive
    false_negative = gt_sum - true_positive
    
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    
    return iou, precision, recall


def img_to_pcd_nuscenes_torch(range_img,
                               maximum_range: float = 100.0,
                               flip_vertical: bool = True,
                               eval: bool = True) -> np.ndarray:
    """PyTorch vectorized implementation (batched or single image)"""
    # Handle single image input
    if isinstance(range_img, np.ndarray):
        if range_img.ndim == 2:
            range_img = range_img[np.newaxis, ...]
        elif range_img.ndim == 3 and range_img.shape[2] == 1:
            range_img = range_img[..., 0]
        range_img = torch.from_numpy(range_img).float()
    else:
        if range_img.ndim == 2:
            range_img = range_img.unsqueeze(0)
        elif range_img.ndim == 3 and range_img.shape[2] == 1:
            range_img = range_img[..., 0]
        range_img = range_img.float()
    
    N, H, W = range_img.shape
    device = range_img.device
    
    # Denormalize if needed
    R = range_img.clone()
    if eval and maximum_range != 1.0:
        R *= maximum_range
    
    # Create index grids (vectorized for all batches)
    rr = torch.arange(H, device=device).repeat_interleave(W)  # Shape: (H*W,)
    cc = torch.arange(W, device=device).repeat(H)              # Shape: (H*W,)
    
    # Undo vertical flip if needed
    row_idx = (H - 1 - rr) if flip_vertical else rr
    
    # Get elevation angles (same for all batches)
    el = ELEV_DEG_PER_RING_NUCSENES_RAD_TORCH.to(device)[row_idx]  # Shape: (H*W,)
    az = ((cc.float() + 0.5) / W) * (2 * torch.pi) - torch.pi  # Shape: (H*W,)
    
    # Extract range data using indexing, broadcasting across batch dimension
    r = R[:, rr, cc]  # Shape: (N, H*W)
    
    # Vectorized coordinate conversion
    cos_el = torch.cos(el)  # Shape: (H*W,)
    sin_el = torch.sin(el)  # Shape: (H*W,)
    cos_az = torch.cos(az)  # Shape: (H*W,)
    sin_az = torch.sin(az)  # Shape: (H*W,)
    
    # Broadcast to batch dimension: (N, 1) * (H*W,) -> (N, H*W)
    x = r * cos_el.unsqueeze(0) * cos_az.unsqueeze(0)
    y = r * cos_el.unsqueeze(0) * sin_az.unsqueeze(0)
    z = r * sin_el.unsqueeze(0)
    
    # Stack to shape (N, H*W, 3)
    pc = torch.stack([x, y, z], dim=2)
    
    return pc.float()

def img_to_pcd_nuscenes(range_img,
                        maximum_range: float = 100.0,
                        flip_vertical: bool = True,
                        eval: bool = True) -> np.ndarray:
    """
    Simplest consistent inverse for HDL-32E NuScenes range images.
    range_img: (H, W) 32x1024
    flip_vertical: wheter the range image top is highest elevation
    eval: is this being called for evaluation? if so denormalize the range image from 0-1 to 0-maximum_range
    """
    if range_img.ndim == 3 and range_img.shape[2] == 1:
        range_img = range_img[..., 0]
    H, W = range_img.shape

    # Valid pixels
    R = range_img.astype(np.float32, copy=True)
    if eval:
        R *= maximum_range # eval 
    rr = np.repeat(np.arange(H), W) # [000000,111111,222222,333333,444444,555555,666666,777777]
    cc = np.tile(np.arange(W), H) # [01234567,01234567,01234567,01234567,01234567,01234567,01234567,01234567]

    # Undo vertical flip from forward pass if needed
    row_idx = (H - 1 - rr) if flip_vertical else rr

    # Elevation directly from row index
    el = ELEV_DEG_PER_RING_NUCSENES_RAD[row_idx]
    az = ((cc.astype(np.float32) + 0.5) / W) * (2*np.pi) - np.pi
    r = R[rr, cc]

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    pc = np.stack([x, y, z], axis=1)
    return pc.astype(np.float32)
    
def img_to_pcd_durlar(img_range, maximum_range = 120):  # 1 x H x W cuda torch
    rows, cols = img_range.shape[:2]
    uu, vv = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    uvs = np.stack((uu, vv), axis=-1).reshape(-1, 2)

    points = np.zeros((rows*cols, 3))
    indices = idx_from_px(uvs, cols)
    points_all = px_to_xyz(uvs, img_range.transpose().reshape(-1) * maximum_range, cols)

    points[indices, :] = points_all
    return points

def img_to_pcd_kitti(img_range, maximum_range = 120, low_res = False, intensity = None):
    if low_res:
        image_rows = 16
    else:
        image_rows = 64
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols

    rowList = []
    colList = []
    for i in range(image_rows):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))


    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi


    lengthList = img_range.reshape(image_rows*image_cols) * maximum_range

    x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
    y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
    z = np.sin(verticalAngle) * lengthList
    if intensity is not None:
        intensity = intensity.reshape(image_rows*image_cols)
        points = np.column_stack((x,y,z,intensity))
    else:    
        points = np.column_stack((x,y,z))

    return points


def img_to_pcd_carla(img_range, maximum_range = 80):
    # img_range = np.flip(img_range)
    rows, cols = img_range.shape[:2]

    v_dir = np.linspace(start=-15, stop=15, num=rows)
    h_dir = np.linspace(start=-180, stop=180, num=cols, endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(rows):
        v_angles = np.append(v_angles, np.ones(cols) * v_dir[i])
        h_angles = np.append(h_angles, h_dir)

    angles = np.stack((v_angles, h_angles), axis=-1).astype(np.float32)
    angles = np.deg2rad(angles)

    r = img_range.flatten() * maximum_range


    x = np.sin(angles[:, 1]) * np.cos(angles[:, 0]) * r
    y = np.cos(angles[:, 1]) * np.cos(angles[:, 0]) * r
    z = np.sin(angles[:, 0]) * r

    points = np.stack((x, y, z), axis=-1)

    return points


def mean_absolute_error(pred_img, gt_img):
    abs_error = (pred_img - gt_img).abs()

    return abs_error.mean()

def chamfer_distance_gpu(points1, points2, num_points = None):
    source = points1[None, :]
    target = points2[None, :]

    chd = chamfer_dist()
    dist1, dist2, _, _ = chd(source, target)
    cdist = (torch.mean(dist1)) + (torch.mean(dist2)) if num_points is None else (dist1.sum()/num_points) + (dist2.sum()/num_points)

    return cdist.detach().cpu()

def chamfer_distance(points1, points2, num_points = None):
    source = torch.from_numpy(points1[None, :]).cuda()
    target = torch.from_numpy(points2[None, :]).cuda()


    chd = chamfer_dist()
    dist1, dist2, _, _ = chd(source, target)
    cdist = (torch.mean(dist1)) + (torch.mean(dist2)) if num_points is None else (dist1.sum()/num_points) + (dist2.sum()/num_points)

    return cdist.detach().cpu()

def depth_wise_unconcate(imgs): # H W
    b, c, h, w = imgs.shape
    new_imgs = torch.zeros((b, h*c, w)).cuda()
    low_res_indices = [range(i, h*c+i, c) for i in range(c)]


    for i, indices in enumerate(low_res_indices):
        new_imgs[:, indices,:] = imgs[:, i, :, :]

    return new_imgs.reshape(b, 1, h*c, w)


def voxelize_point_cloud(point_cloud, grid_size, min_coord, max_coord):
    # Calculate the dimensions of the voxel grid
    dimensions = ((max_coord - min_coord) / grid_size).astype(int) + 1

    # Create the voxel grid
    voxel_grid = np.zeros(dimensions, dtype=bool)

    # Assign points to voxels
    indices = ((point_cloud - min_coord) / grid_size).astype(int)
    voxel_grid[tuple(indices.T)] = True

    return voxel_grid

def calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth):
    
    intersection = np.logical_and(voxel_grid_predicted, voxel_grid_ground_truth)
    union = np.logical_or(voxel_grid_predicted, voxel_grid_ground_truth)

    iou = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(intersection)
    false_positive = np.sum(voxel_grid_predicted) - true_positive
    false_negative = np.sum(voxel_grid_ground_truth) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return iou, precision, recall

def inverse_huber_loss(output, target):
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C))
