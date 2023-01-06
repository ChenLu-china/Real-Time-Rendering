# Cuda C++

## Basic Cuda Programming Theory

Set code run on which kind of device:

```bash
__global__, __device__, __host__
```
__Grid__,   __Block__,   __Thread__


## Advanced Cuda Extension Programming

Advice to download Libtorch from Pytorch offical websit, imitate code and processing from offcial examples. Hit: there is no effecient method to debug .cu and .cuh scripts, except you directly develop CUDA C or CUDA C++ code with nvcc or g++ compiler tools. JIT mode also need to wait for installing cuda extensions using setuptools.
