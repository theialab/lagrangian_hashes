import os
import sys
from setuptools import setup, find_packages, dist
import glob
import logging
import subprocess

import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

PACKAGE_NAME = "laghash"
URL = "https://github.com/theialab/lagrangian_hashes"
LICENSE = "MIT"
version = "0.1.0"


def get_extensions():
    extra_compile_args = {"cxx": ["-O3", "-fdiagnostics-color=always"]}
    define_macros = []
    include_dirs = []
    extensions = []
    sources = glob.glob("csrc/**/*.cpp", recursive=True)

    if len(sources) == 0:
        print("No source files found for extension, skipping extension compilation")
        return None

    if torch.cuda.is_available():
        define_macros += [
            ("WITH_CUDA", None),
            ("THRUST_IGNORE_CUB_VERSION_CHECK", None),
        ]
        sources += glob.glob("csrc/**/*.cu", recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({"nvcc": ["-O3", "-maxrregcount=40"]})
        # include_dirs = get_include_dirs()
    else:
        raise Exception("CUDA not available")

    extensions.append(
        extension(
            name="_C",
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    )

    for ext in extensions:
        ext.libraries = ["cudart_static" if x == "cudart" else x for x in ext.libraries]

    return extensions


if __name__ == "__main__":
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author="Shrisudhan",
        url=URL,
        license=LICENSE,
        python_requires=">=3.8",
        # Package info
        packages=find_packages(),
        include_package_data=True,
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    )
