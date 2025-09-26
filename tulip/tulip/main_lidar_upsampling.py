# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import pdb
from mmcv import Config

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util.datasets import generate_dataset, collate_fn

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import model.tulip as tulip
from model.CMTulip import CMTULIP, freeze_multiview_backbone, unfreeze_multiview_backbone
import wandb
from util.wandb_artifact import WandbArtifactHook


def load_config(args):
    """Load configuration from YAML file using MMCV Config"""
    config_path = args.config
    if config_path and os.path.exists(config_path):
        config = Config.fromfile(config_path)
        config.eval = args.eval
        return config
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def get_config_value(config, *keys, default=None):
    """Get nested config value with fallback to default"""
    current = config
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # Only config file argument
    parser.add_argument('--config', type=str, required=True,
                        help='path to YAML config file')
    parser.add_argument('--eval', action='store_true', help="evaluation")

    return parser

def build_backbone_split_param_groups(
    model: torch.nn.Module,
    base_param_groups,
    *,
    backbone_attr: str = "multiview_backbone",
    phase: int,
    phase_one_lr_scale: float = 0.0,
    phase_one_weight_decay: float = 0.0,
    phase_two_lr_scale: float = 0.1,
    phase_two_weight_decay: float = 0.005,
):
    """Return optimizer param groups where backbone params are separated and adjusted.

    Parameters
    - model: model that contains the backbone under `backbone_attr`.
    - base_param_groups: param groups produced by timm's optim factory.
    - backbone_attr: attribute name of the backbone in model/module.
    - phase: 1 or 2; controls how backbone group's lr/weight_decay are set.
    - phase_one_lr_scale / phase_one_weight_decay: overrides for phase 1 (frozen bb).
    - phase_two_lr_scale / phase_two_weight_decay: overrides for phase 2 (unfrozen bb).
    """
    backbone_module = getattr(model, backbone_attr)
    backbone_param_ids = {id(p) for p in backbone_module.parameters()}

    split_param_groups = []
    for group in base_param_groups:
        params_backbone = [p for p in group["params"] if id(p) in backbone_param_ids]
        params_non_backbone = [p for p in group["params"] if id(p) not in backbone_param_ids]

        if len(params_non_backbone) > 0:
            new_group_non_backbone = {k: v for k, v in group.items() if k != "params"}
            new_group_non_backbone["params"] = params_non_backbone
            split_param_groups.append(new_group_non_backbone)

        if len(params_backbone) > 0:
            new_group_backbone = {k: v for k, v in group.items() if k != "params"}
            new_group_backbone["params"] = params_backbone
            if phase == 1:
                new_group_backbone["lr_scale"] = phase_one_lr_scale
                new_group_backbone["weight_decay"] = phase_one_weight_decay
            else:
                # keep any existing lr_scale, but apply multiplicative scale for backbone
                new_group_backbone["lr_scale"] = new_group_backbone.get("lr_scale", 1.0) * phase_two_lr_scale
                new_group_backbone["weight_decay"] = phase_two_weight_decay
            split_param_groups.append(new_group_backbone)

    return split_param_groups

def main(args):
    # Load configuration from YAML file using MMCV Config
    args = get_args_parser()
    args = args.parse_args()

    config = load_config(args)
    
    collate_func = None
    if config.model_select == "CMTULIP":
        from engine_upsampling_w_image import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
        im2col_step = get_config_value(config, 'batch_size') * 6 # assuming cam 6 views
        config.im2col_step = im2col_step
        
        collate_func = collate_fn
    else:
        from engine_upsampling import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
    
    # Add datetime timestamp to output_dir to ensure each run has distinct space
    if config.output_dir and not args.eval:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_output_dir = config.output_dir
        config.output_dir = f"{original_output_dir}_{timestamp}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {config.output_dir}")
    elif config.output_dir and args.eval:
        # For evaluation, check if it's a specific checkpoint file or directory
        if config.output_dir.endswith("pth"):
            # If it's a checkpoint file, use its directory as output_dir
            config.resume = config.output_dir
            config.output_dir = os.path.dirname(config.output_dir)
            print(f"Evaluation mode - checkpoint file: {config.resume}")
            print(f"Evaluation mode - output directory: {config.output_dir}")
        else:
            # If it's a directory, use it as-is
            print(f"Evaluation mode - using output directory: {config.output_dir}")
    
    # Use MMCV Config object directly - no intermediate args variable needed
    misc.init_distributed_mode(config)


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(config).replace(', ', ',\n'))

    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = generate_dataset(is_train = True, args = config)
    dataset_val = generate_dataset(is_train = False, args = config)

    print(f"There are totally {len(dataset_train)} training data and {len(dataset_val)} validation data")


    
    if True:  # config.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        
        # Validation set uses only one rank to write the summary
        sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Logger is only used in one rank
    wandb_artifact_hook = None
    if global_rank == 0:
        if config.wandb_disabled:
            mode = "disabled"
        else:
            mode = "online"
        project_name = f"{config.project_name}_eval" if args.eval else config.project_name
        wandb.init(project=project_name,
                    entity=config.entity,
                    name = config.run_name, 
                    mode=mode,
                    sync_tensorboard=True)
        wandb.config.update(config)
        
        # Initialize WandbArtifactHook for automatic checkpoint uploading
        if not config.wandb_disabled:
            wandb_artifact_hook = WandbArtifactHook(
                wandb_entity=config.entity,
                wandb_project=project_name,
                dir_path=config.output_dir
            )
            wandb_artifact_hook.before_run(output_dir=config.output_dir, logger=None)
    if global_rank == 0 and config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=config.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True
    )


    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=False,
        collate_fn=collate_func
    )
    
    
    # define the model
    if config.model_select == "CMTULIP":
        ef_bs = config.batch_size * config.accum_iter * misc.get_world_size() * 6
        print(f"Multiview effective batch size: {ef_bs}")
        model = CMTULIP(
            backbone_config=config.backbone,
            img_size=tuple(config.img_size_low_res),
            target_img_size=tuple(config.img_size_high_res),
            patch_size=tuple(config.patch_size),
            im2col_step=config.im2col_step,
            in_chans=config.in_chans,
            window_size=config.window_size,
            swin_v2=config.swin_v2,
            pixel_shuffle=config.pixel_shuffle,
            circular_padding=config.circular_padding,
            log_transform=config.log_transform,
            patch_unmerging=config.patch_unmerging,
            lss_weights_path=config.lss_weights_path
        )
    else:
        model = tulip.__dict__[config.model_select](
            img_size=tuple(config.img_size_low_res),
            target_img_size=tuple(config.img_size_high_res),
            patch_size=tuple(config.patch_size), 
            in_chans=config.in_chans,
            window_size=config.window_size,
            swin_v2=config.swin_v2,
            pixel_shuffle=config.pixel_shuffle,
            circular_padding=config.circular_padding,
            log_transform=config.log_transform,
            patch_unmerging=config.patch_unmerging
        )
    
    if args.eval and os.path.exists(config.output_dir):
        print("Loading Checkpoint and directly start the evaluation")
        # Checkpoint file handling is already done above
        if not config.output_dir.endswith("pth"):
            get_latest_checkpoint(config)
        misc.load_model(
                args=config, model_without_ddp=model, optimizer=None,
                loss_scaler=None)
        model.to(device)
        
        print("Start Evaluation")
        if config.mc_drop:
            print("Evaluation with Monte Carlo Dropout")
            MCdrop(data_loader_val, model, device, log_writer = log_writer, args = config)
        else:
            evaluate(data_loader_val, model, device, log_writer = log_writer, args = config)
        print("Evaluation finished")


        exit(0)
    else:
        print("Start Training")
        

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = config.batch_size * config.accum_iter * misc.get_world_size()
    
    if config.lr is None:  # only base_lr is specified
        config.lr = config.blr * eff_batch_size / 256

    print("base lr: %.2e" % (config.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % config.lr)

    print("accumulate grad iterations: %d" % config.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if config.model_select == "CMTULIP":
        if not args.eval:
            model.train(True)
            freeze_multiview_backbone(model)
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = optim_factory.param_groups_layer_decay(model_without_ddp, config.weight_decay)
    if config.model_select == "CMTULIP":
        # Configure defaults (can be overridden by config)
        config.backbone_lr_scale = getattr(config, 'backbone_lr_scale', 0.1)
        config.backbone_weight_decay = getattr(config, 'backbone_weight_decay', 0.005)

        param_groups = build_backbone_split_param_groups(
            model_without_ddp,
            param_groups,
            backbone_attr="multiview_backbone",
            phase=1,
            phase_one_lr_scale=0.0,
            phase_one_weight_decay=0.0,
            phase_two_lr_scale=config.backbone_lr_scale,
            phase_two_weight_decay=config.backbone_weight_decay,
        )

    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    

    misc.load_model(args=config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {config.epochs} epochs")
    start_time = time.time()
    phase_one_epochs = getattr(config, "phase_one_epochs", int(0.2 * config.epochs))
    print(f"Phase one training for {phase_one_epochs} epochs")
    two_phase_train = (config.model_select == "CMTULIP") and (not args.eval)
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if two_phase_train and epoch >= phase_one_epochs:
            print("Second Phase Training: Unfreeze the multiview backbone")
            unfreeze_multiview_backbone(model, train_bn=True)
            if config.distributed:
                torch.cuda.synchronize()
                if misc.get_world_size() > 1:
                    torch.distributed.barrier()
                model = torch.nn.parallel.DistributedDataParallel(
                    model.module, device_ids=[config.gpu], find_unused_parameters=False
                )
                model_without_ddp = model.module
            else:
                # single-GPU: keep model as is
                model_without_ddp = model
            # Rebuild and split param groups for phase 2
            param_groups = optim_factory.param_groups_layer_decay(model.module, config.weight_decay)
            param_groups = build_backbone_split_param_groups(
                model.module,
                param_groups,
                backbone_attr="multiview_backbone",
                phase=2,
                phase_one_lr_scale=0.0,
                phase_one_weight_decay=0.0,
                phase_two_lr_scale=config.backbone_lr_scale,
                phase_two_weight_decay=config.backbone_weight_decay,
            )

            optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
            optimizer.zero_grad(set_to_none=True)
            two_phase_train = False
            
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=config
        )
        if config.output_dir and (epoch % config.save_frequency == 0 or epoch + 1 == config.epochs):
            misc.save_model(
                args=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
            # Use WandbArtifactHook to upload checkpoint as wandb artifact
            if global_rank == 0 and wandb_artifact_hook is not None:
                try:
                    wandb_artifact_hook.after_train_epoch()
                    print(f"Uploaded checkpoint artifacts for epoch {epoch}")
                except Exception as e:
                    print(f"Warning: Failed to upload checkpoint as wandb artifact: {e}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if config.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(config.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print('Training finished')

    if global_rank == 0:
        wandb.finish()
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    
    main(args)
