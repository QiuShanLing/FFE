from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
from pathlib import Path

ext_modules = [
    Pybind11Extension(
        "ffe._parser",
        ["cpp/parser.cpp"],
    ),
]

setup(
    name="ffe",
    ext_modules=ext_modules,
)
