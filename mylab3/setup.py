from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='linear_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='linear_cuda_extension',  
            sources=['custom_cuda.cpp', 'custom.cu'],  
            extra_compile_args={
                'cxx': ['-O3'],  
                'nvcc': ['-O3', '-lineinfo']  
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
