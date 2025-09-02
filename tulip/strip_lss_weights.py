#!/usr/bin/env python3
"""
Extract LSS (Lift-Splat-Shoot) module weights from a pretrained BEVDepth model.

This script loads a full BEVDepth checkpoint and saves only the LSS backbone weights
that are needed for the CMTULIP model's multiview_backbone component.
"""

import torch
import argparse
import os
from pathlib import Path


def extract_lss_weights(checkpoint_path, output_path, model_name="model.backbone"):
    """
    Extract LSS weights from BEVDepth checkpoint.
    
    Args:
        checkpoint_path: Path to the full BEVDepth checkpoint (.pth file)
        output_path: Path to save the extracted LSS weights
        model_name: Model name to look for in the checkpoint
    """
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    print(f"Total keys in checkpoint: {len(state_dict)}")
    
    # LSS-related keys to extract
    lss_keys = []
    
    # Common LSS backbone patterns in BEVDepth
    lss_patterns = [
        'img_backbone',           # Image backbone (ResNet, etc.)
        'img_neck',              # FPN neck
        'depth_net',             # Depth prediction network
        'backbone',              # Alternative backbone naming
    ]
    
    # Extract keys that match LSS patterns
    for key in state_dict.keys():
        # Remove model name prefix if present
        clean_key = key
        if key.startswith(f'{model_name}.'):
            clean_key = key[len(f'{model_name}.'):]
            print(f"Cleaned key: {clean_key}")
        
        # Check if key matches any LSS pattern
        for pattern in lss_patterns:
            if pattern in clean_key:
                lss_keys.append(key)
                break
    
    print(f"Found {len(lss_keys)} LSS-related keys")
    
    if not lss_keys:
        print("Warning: No LSS keys found. Available keys:")
        for i, key in enumerate(list(state_dict.keys())[:20]):  # Show first 20 keys
            print(f"  {i+1:2d}: {key}")
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more")
        return False
    
    # Create filtered state dict
    lss_state_dict = {}
    for key in lss_keys:
        # Remove model name prefix for cleaner keys
        clean_key = key
        if key.startswith(f'{model_name}.'):
            clean_key = key[len(f'{model_name}.'):]
        
        lss_state_dict[clean_key] = state_dict[key]
        print(f"Extracted: {clean_key} -> {state_dict[key].shape}")
    
    # Save the extracted weights
    output_data = {
        'state_dict': lss_state_dict,
        'model_name': model_name,
        'extracted_keys': len(lss_keys),
        'total_original_keys': len(state_dict)
    }
    
    print(f"\nSaving {len(lss_state_dict)} LSS weights to: {output_path}")
    torch.save(output_data, output_path)
    
    # Calculate size reduction
    original_size = sum(p.numel() for p in state_dict.values())
    lss_size = sum(p.numel() for p in lss_state_dict.values())
    reduction = (1 - lss_size / original_size) * 100
    
    print(f"Size reduction: {reduction:.1f}% ({lss_size:,} / {original_size:,} parameters)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract LSS weights from BEVDepth checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to BEVDepth checkpoint (.pth file)")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Output path for LSS weights (default: checkpoint_name_lss.pth)")
    parser.add_argument("-m", "--model-name", type=str, 
                       default="bev_depth_lss_r50_256x704_128x128_24e_2key_ema",
                       help="Model name in checkpoint to look for")
    parser.add_argument("--list-keys", action="store_true",
                       help="List all keys in checkpoint and exit")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    # Set default output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_lss.pth"
    else:
        output_path = args.output
    
    # List keys if requested
    if args.list_keys:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"All keys in checkpoint ({len(state_dict)} total):")
        for i, key in enumerate(state_dict.keys()):
            print(f"  {i+1:3d}: {key}")
        return 0
    
    # Extract LSS weights
    success = extract_lss_weights(args.checkpoint, output_path)
    
    if success:
        print(f"\n✅ Successfully extracted LSS weights to: {output_path}")
        return 0
    else:
        print(f"\n❌ Failed to extract LSS weights")
        return 1


if __name__ == "__main__":
    exit(main())
