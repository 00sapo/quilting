import numpy
from Cython.Build import cythonize
from setuptools import Extension

ext_modules = [
    Extension(
        "_quilting",
        ["src/quilting/_quilting.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp", "-Ofast"],
        extra_link_args=["-fopenmp", "-Ofast"],
    )
]


def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(ext_modules=cythonize(ext_modules))
    return setup_kwargs
