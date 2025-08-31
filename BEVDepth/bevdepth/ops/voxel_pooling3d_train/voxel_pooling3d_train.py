# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

from . import voxel_pooling_train_ext


class VoxelPooling3DTrain(Function):

    @staticmethod
    def forward(ctx,
                geom_xyz: torch.Tensor,        # [B, N, 3] voxel indices (x,y,z) per sample
                input_features: torch.Tensor,  # [B, N, C]
                depth_logits: torch.Tensor,   # [B, N]
                voxel_num: torch.Tensor        # [3] -> [X, Y, Z]
                ) -> torch.Tensor:
        """3D voxel pooling forward (keeps Z; no height compression).

        Args:
            geom_xyz: (B, N, 3) integer voxel indices in ego frame (x,y,z).
                      Invalid samples should be set out-of-range; we will mask them.
            input_features: (B, N, C) features per lifted sample.
            voxel_num: (3,) number of voxels in X, Y, Z.

        Returns:
            (B, C, Z, Y, X) fused 3D feature volume.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        assert depth_logits.is_contiguous()

        # we never backprop through geom indices
        ctx.mark_non_differentiable(geom_xyz)

        # prepare grad buffer for backward (same shape as input_features)
        grad_input_features = torch.zeros_like(input_features)

        # flatten batch-wise to match kernel interface
        B = input_features.shape[0]
        geom_xyz = geom_xyz.reshape(B, -1, geom_xyz.shape[-1]).contiguous()
        input_features = input_features.reshape(B, -1, input_features.shape[-1]).contiguous()
        N = input_features.shape[1]
        C = input_features.shape[2]

        # voxel dims
        X = int(voxel_num[0])
        Y = int(voxel_num[1])
        Z = int(voxel_num[2])

        # kernel expects (B, Z, Y, X, C) contiguous, channel last
        output_features = input_features.new_zeros(B, Z, Y, X, C)
        uni_depth_logits_output = input_features.new_zeros(B, Z, Y, X)

        # Save the (b,z,y,x) voxel for each input point; -1 marks invalid
        pos_memo = geom_xyz.new_full((B, N, 4), -1)

        # NOTE: call your updated 3D wrapper
        # If you kept the old name, rename here accordingly.
        voxel_pooling_train_ext.voxel_pooling3d_train_forward_wrapper(
            B, N, C, X, Y, Z,
            geom_xyz,          # [B, N, 3] (x,y,z)
            input_features,     # [B, N, C]
            depth_logits,       # [B, N]
            output_features,    # [B, Z, Y, X, C]
            uni_depth_logits_output, # [B, Z, Y, X]
            pos_memo,           # [B, N, 4] (b,z,y,x) or -1
        )

        # stash tensors for backward
        ctx.save_for_backward(grad_input_features, pos_memo)
        #TODO: save depth_logits for backward ( DONT KNOW IF IT IS NEEDED )
        # ctx.save_for_backward(depth_logits)

        # return channel-first 3D volume
        return output_features.permute(0, 4, 1, 2, 3).contiguous(), uni_depth_logits_output.contiguous()  # (B, C, Z, Y, X)

    @staticmethod
    def backward(ctx, grad_output_features):
        # grad_output_features: (B, C, Z, Y, X)
        grad_input_features, pos_memo = ctx.saved_tensors
        B, C, Z, Y, X = grad_output_features.shape

        # valid samples
        kept = (pos_memo[..., 0] >= 0)

        # flatten input-feat grad for easy indexing
        gi_shape = grad_input_features.shape  # (B, N, C)
        gi = grad_input_features.view(B, -1, C)

        # gather grads from voxel volume back to each input sample
        b = pos_memo[..., 0].long()
        z = pos_memo[..., 1].long()
        y = pos_memo[..., 2].long()
        x = pos_memo[..., 3].long()

        gi[kept] = grad_output_features[b[kept], :, z[kept], y[kept], x[kept]]

        return None, grad_input_features.view(gi_shape), None


voxel_pooling3d_train = VoxelPooling3DTrain.apply
