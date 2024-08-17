from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# python setup.py build_ext --inplace
setup(
    name='custom_cpp_extention',
    version='0.1',
    author='Yuan Jiaqi',
    author_email='2460058215@qq.com',
    description='A custom C++ extension for PyTorch',
    ext_modules=[
        CppExtension(
            name='mycpp_extention',
            sources=['custom_linear.cpp','custom_conv.cpp','custom_dropout2d.cpp','bindings.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

