
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='luchen',
    ext_modules=[
        CUDAExtension('luchen', [
            '/home/chenlu/projects/chenlu/cuda_c++/cuda/csrc/svox2.cpp',
            '/home/chenlu/projects/chenlu/cuda_c++/cuda/csrc/svox2_kernel.cu',
            '/home/chenlu/projects/chenlu/cuda_c++/cuda/csrc/misc_kernel.cu',
            '/home/chenlu/projects/chenlu/cuda_c++/cuda/csrc/render_lerp_kernel_cuvol.cu',
            '/home/chenlu/projects/chenlu/cuda_c++/cuda/csrc/rende_depth_kernel.cu'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
