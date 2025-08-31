# TULIP Configuration System

This directory contains YAML configuration files for the TULIP LiDAR upsampling system. The configuration system allows you to manage all model-related parameters in a structured, readable format.

## Configuration Files

### `model_config.yaml`
The main configuration file containing all default parameters. This file serves as a template and can be customized for different experiments.

### `example_kitti_config.yaml`
An example configuration file specifically for KITTI dataset experiments, demonstrating how to customize parameters for different use cases.

## Configuration Structure

The configuration is organized into logical sections:

### Model Configuration (`model`)
- Model selection (`tulip_base`, `tulip_large`, `CMTULIP`)
- Window and patch settings
- Upsampling head options
- Swin Transformer version selection

### Backbone Configuration (`backbone`)
- Spatial boundaries (x_bound, y_bound, z_bound, d_bound)
- Output dimensions and channels
- Image backbone and neck configurations
- Depth network parameters

### Dataset Configuration (`dataset`)
- Dataset selection (`durlar`, `carla`, `kitti`)
- Image dimensions (low and high resolution)
- Data paths
- Processing options

### Training Configuration (`training`)
- Batch size and epochs
- Learning rate settings
- Augmentation options
- Checkpoint and logging settings

### Evaluation Configuration (`evaluation`)
- Evaluation mode settings
- Monte Carlo dropout parameters
- Grid size for voxelization

## Usage

### Basic Usage
```bash
python main_lidar_upsampling.py --config ./configs/model_config.yaml
```

### Custom Configuration
```bash
python main_lidar_upsampling.py --config ./configs/example_kitti_config.yaml
```

### Using Bash Scripts
```bash
# Training
./bash_scripts/tulip_upsampling_kitti.sh

# Evaluation
./bash_scripts/tulip_evaluation_kitti.sh
```

## Configuration-Driven Design

The system is now fully configuration-driven:
1. All parameters are defined in YAML configuration files
2. No command line argument overrides needed
3. Easy to maintain reproducible experiments
4. Simple to switch between different configurations

## Creating Custom Configurations

1. Copy an existing config file
2. Modify the parameters for your specific use case
3. Save with a descriptive name
4. Use with the `--config` argument

## Example: KITTI Dataset

```yaml
dataset:
  select: 'kitti'
  img_size_low_res: [32, 1024]
  img_size_high_res: [128, 1024]
  data_path_low_res: './data/KITTI/train'
  data_path_high_res: './data/KITTI/train'
```

## Example: CARLA Dataset

```yaml
dataset:
  select: 'carla'
  img_size_low_res: [64, 2048]
  img_size_high_res: [256, 2048]
  data_path_low_res: './data/CARLA/train'
  data_path_high_res: './data/CARLA/train'
```

## Dependencies

Make sure to install MMCV 1.7.0:
```bash
pip install 'mmcv>=1.7.0,<2.0.0'
```

## Migration from Hardcoded Values

If you were previously using hardcoded values in the script:
1. Copy the relevant parameters to a YAML config file
2. Use the `--config` argument to specify your config
3. All parameters are now managed through configuration files

## Benefits

- **Reproducibility**: Config files can be version controlled
- **Flexibility**: Easy to switch between different configurations
- **Maintainability**: Centralized parameter management
- **Collaboration**: Team members can share and modify configs
- **Experimentation**: Quick parameter tuning without code changes
- **MMCV Integration**: Uses MMCV's robust config system with dot notation access
- **Dot Notation Access**: Clean access to nested config values (e.g., `config.model.select`)
- **Type Safety**: Better error handling and validation
- **Config Validation**: MMCV provides better error messages for missing or invalid config values
- **Direct Config Usage**: Passes MMCV config object directly to all functions without intermediate variables
