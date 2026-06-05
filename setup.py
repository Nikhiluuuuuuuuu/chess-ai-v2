from setuptools import setup, Extension
import pybind11
import sys

# Default to GCC/Clang flags
compile_args = ['-O3', '-std=c++17', '-ffast-math']

# If Windows (MSVC), use Microsoft compiler flags
if sys.platform == 'win32':
    compile_args = ['/O2', '/std:c++17', '/fp:fast']

ext_modules = [
    Extension(
        'mcts_ext',
        ['mcts_ext.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=compile_args
    ),
]

setup(
    name='mcts_ext',
    version='1.0',
    description='C++ Batched MCTS Extension using bitboards for DeepVisionElite',
    ext_modules=ext_modules,
)
