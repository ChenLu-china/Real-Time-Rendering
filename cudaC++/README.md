# Cuda C++

## Basic Cuda Programming Theory

Set code run on which kind of device:

```bash
__global__, __device__, __host__
```
__Grid__,   __Block__,   __Thread__

GPU 组织架构：

SM(stream multiprocessor): each grid run on one SM  
  
WARP: The smallest unit of SM. Normally, the size of a WARP is 32, because a WARP means a group of 32 threads. The effect of WARP is to aviod

BLOCK：The definiation of BLOCK is same as WARP, but the size of BLOCK not fixed, you can set it flexiable, normally, the times of 32.

THREAD: every code run on thread, and the processor of code are same except WARP.

## Advanced Cuda Extension Programming

Advice to download Libtorch from Pytorch offical websit, imitate code and processing from offcial examples. Hit: there is no effecient method to debug .cu and .cuh scripts, except you directly develop CUDA C or CUDA C++ code with nvcc or g++ compiler tools. JIT mode also need to wait for installing cuda extensions using setuptools.

