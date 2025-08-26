// Copyright (c) Megvii Inc. All rights reserved.
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_BLOCK_X 32
#define THREADS_BLOCK_Y 4
#define THREADS_PER_BLOCK (THREADS_BLOCK_X * THREADS_BLOCK_Y)
#define DIVUP(m, n) (((m) / (n)) + ((m) % (n) > 0))

// When __half atomicAdd isn't available on your arch, you need a fallback.
// (Optional) Add your own atomicAdd for half here if you target < sm_70.

template <typename T>
__global__ void voxel_pooling3d_train_forward_kernel(
    int batch_size, int num_points, int num_channels,
    int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int* __restrict__ geom_xyz,         // (B * N, 3) -> x,y,z voxel indices per sample
    const T*  __restrict__ input_features,    // (B * N, C)
    float*       __restrict__ output_features,   // (B, Z, Y, X, C) flattened
    int*      __restrict__ pos_memo           // (B * N, 4): b, z, y, x  (or left unset if invalid)
) {
  const int bidx = blockIdx.x;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  const int sample_dim = THREADS_PER_BLOCK;
  const int idx_in_block = tidy * THREADS_BLOCK_X + tidx;

  const int block_sample_idx  = bidx * sample_dim;
  const int thread_sample_idx = block_sample_idx + idx_in_block;

  const int total_samples = batch_size * num_points;

  __shared__ int geom_xyz_shared[THREADS_PER_BLOCK * 3];

  // stage 1: load geom indices to shared and prefill pos_memo for valid entries
  if (thread_sample_idx < total_samples) {
    const int gx = geom_xyz[thread_sample_idx * 3 + 0];
    const int gy = geom_xyz[thread_sample_idx * 3 + 1];
    const int gz = geom_xyz[thread_sample_idx * 3 + 2];

    geom_xyz_shared[idx_in_block * 3 + 0] = gx;
    geom_xyz_shared[idx_in_block * 3 + 1] = gy;
    geom_xyz_shared[idx_in_block * 3 + 2] = gz;

    if ((gx >= 0 && gx < num_voxel_x) &&
        (gy >= 0 && gy < num_voxel_y) &&
        (gz >= 0 && gz < num_voxel_z)) {
      const int b = thread_sample_idx / num_points;
      pos_memo[thread_sample_idx * 4 + 0] = b;
      pos_memo[thread_sample_idx * 4 + 1] = gz;
      pos_memo[thread_sample_idx * 4 + 2] = gy;
      pos_memo[thread_sample_idx * 4 + 3] = gx;
    } else {
      // mark invalid (optional, useful for backward)
      pos_memo[thread_sample_idx * 4 + 0] = -1;
      pos_memo[thread_sample_idx * 4 + 1] = -1;
      pos_memo[thread_sample_idx * 4 + 2] = -1;
      pos_memo[thread_sample_idx * 4 + 3] = -1;
    }
  }

  __syncthreads();

  // stage 2: accumulate features into (Z,Y,X) voxels
  for (int i = tidy; i < THREADS_PER_BLOCK && block_sample_idx + i < total_samples; i += THREADS_BLOCK_Y) {
    const int gx = geom_xyz_shared[i * 3 + 0];
    const int gy = geom_xyz_shared[i * 3 + 1];
    const int gz = geom_xyz_shared[i * 3 + 2];

    if (gx < 0 || gx >= num_voxel_x ||
        gy < 0 || gy >= num_voxel_y ||
        gz < 0 || gz >= num_voxel_z) {
      continue;
    }

    const int sample_idx = block_sample_idx + i;
    const int b = sample_idx / num_points;

    // Linear voxel index: ((((b * Z) + z) * Y) + y) * X + x
    const long long voxel_lin =
      ((( (long long)b * num_voxel_z + gz) * num_voxel_y + gy) * num_voxel_x + gx);
    for (int c = tidx; c < num_channels; c += THREADS_BLOCK_X) {
      const long long off = voxel_lin * (long long)num_channels + c;
      atomicAdd(&output_features[off], input_features[(long long)sample_idx * num_channels + c]);
    }
  }
}

void voxel_pooling3d_train_forward_kernel_launcher(
    int batch_size, int num_points, int num_channels,
    int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int* geom_xyz,
    const float* input_features,
    float* output_features,
    int* pos_memo,
    cudaStream_t stream) {

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK));
  dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);

  voxel_pooling3d_train_forward_kernel<float>
      <<<blocks, threads, 0, stream>>>(
          batch_size, num_points, num_channels,
          num_voxel_x, num_voxel_y, num_voxel_z,
          geom_xyz, input_features, output_features, pos_memo);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void voxel_pooling3d_train_forward_kernel_launcher(
    int batch_size, int num_points, int num_channels,
    int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int* geom_xyz,
    const half* input_features,
    float* output_features,
    int* pos_memo,
    cudaStream_t stream) {

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK));
  dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);

  voxel_pooling3d_train_forward_kernel<half>
      <<<blocks, threads, 0, stream>>>(
          batch_size, num_points, num_channels,
          num_voxel_x, num_voxel_y, num_voxel_z,
          geom_xyz, input_features, output_features, pos_memo);

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
