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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.datasets import generate_dataset, collate_fn
from util.pos_embed import interpolate_pos_embed

import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import model.tulip as tulip
from model.CMTulip import CMTULIP
import wandb


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

def main(args):
    # Load configuration from YAML file using MMCV Config
    args = get_args_parser()
    args = args.parse_args()

    config = load_config(args)
    
    if config.model_select == "CMTULIP":
        from engine_upsampling_w_image import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
    else:
        from engine_upsampling import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
    
    if config.output_dir and not args.eval:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
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
    if global_rank == 0:
        if config.wandb_disabled:
            mode = "disabled"
        else:
            mode = "online"
        wandb.init(project=config.project_name,
                    entity=config.entity,
                    name = config.run_name, 
                    mode=mode,
                    sync_tensorboard=True)
        wandb.config.update(config)
    if global_rank == 0 and config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=config.log_dir)
    else:
        log_writer = None


    collate_func = None
    if config.model_select == "CMTULIP":
        collate_func = collate_fn

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=True,
        collate_fn=collate_func
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
      
        model = CMTULIP(
            backbone_config=config.backbone,
            img_size=tuple(config.img_size_low_res),
            depths=config.depths,
            target_img_size=tuple(config.img_size_high_res),
            patch_size=tuple(config.patch_size), 
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
        if config.output_dir.endswith("pth"):
            config.resume = config.output_dir
            config.output_dir = os.path.dirname(config.output_dir)
        else:
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

    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = optim_factory.param_groups_layer_decay(model_without_ddp, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    

    misc.load_model(args=config, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {config.epochs} epochs")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)
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
