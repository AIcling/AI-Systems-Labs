from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_linear_cpp',
    ext_modules=[
        CppExtension('custom_linear_cpp', ['custom_linear.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
