
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= Q) return

#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)

#define CUDA_MAX_THREADS at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock

#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox2.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

// transform world coordinate to grid coordinate
__device__ __inline__ void transform_coord(float * __restrict__ point,
                                          const float* __restrict__ scaling,
                                          const float* __restrict__ offset){

    point[0] = fmaf(point[0], scaling[0], offset[0]);
    point[1] = fmaf(point[1], scaling[1], offset[1]);
    point[2] = fmaf(point[2], scaling[2], offset[2]);
}

__device__ __inline__ static float _rnorm(
                const float* __restrict__ dir) {
    // return 1.f / _norm(dir);
    return rnorm3df(dir[0], dir[1], dir[2]);
}

__host__ __device__ __inline__ static float _dot(
                const float* __restrict__ x,
                const float* __restrict__ y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

// Linear interp
// Subtract and fused multiply-add
// (1-w) a + w b
template<class T>
__host__ __device__ __inline__ T lerp(T a, T b, T w) {
    return fmaf(w, b - a, a);
}

#define int_div2_ceil(x) ((((x) - 1) >> 1) + 1)