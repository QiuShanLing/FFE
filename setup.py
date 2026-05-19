from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "ffe._parser",
        ["cpp/parser.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="ffe",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=ext_modules,
)
