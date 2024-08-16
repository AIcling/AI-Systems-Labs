from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_linear_cpp',
    version='0.1',
    author='Yuan Jiaqi',
    author_email='2460058215@qq.com',
    description='A custom C++ extension for PyTorch',
    ext_modules=[
        CppExtension(
            name='custom_linear_cpp',
            sources=['custom_linear.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

