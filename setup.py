from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'mcts_ext',
        ['mcts_ext.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-ffast-math']
    ),
]

setup(
    name='mcts_ext',
    version='1.0',
    description='C++ Batched MCTS Extension using bitboards for DeepVisionElite',
    ext_modules=ext_modules,
)
