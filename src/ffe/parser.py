from functools import lru_cache
from os import PathLike
from typing import Union

from . import _parser  # type: ignore
from .data import FFData
from .utils import FFEToXarray

PathInput = Union[str, PathLike[str]]


def _as_path_str(path: PathInput) -> str:
    return str(path)


@lru_cache(maxsize=1024)
def _parse_ffe_raw(path: str):
    return _parser.parse_ffe(path)


@lru_cache(maxsize=1024)
def _parse_ffe_array(path: str):
    headers, frequencies, data = _parser.parse_ffe_array(path)
    return tuple(headers), frequencies, data


def parse_ffe(path: PathInput):
    """Parse an FFE file and return the low-level C++ FFEFile object."""
    return _parse_ffe_raw(_as_path_str(path))


def parse_ffe_array(path: PathInput):
    """Parse an FFE file into ``(headers, frequencies, data)``.

    ``data`` has shape ``(Frequency, SpatialPoint, Column)`` and crosses the
    C++/Python boundary once, which is faster than collecting each section.
    """
    return _parse_ffe_array(_as_path_str(path))


def parse_ffe_dataset(path: PathInput):
    """Parse an FFE file and return an xarray Dataset."""
    return FFEToXarray.from_file(path).convert()


def parse(path: PathInput) -> FFData:
    """Parse an FFE file and return the high-level FFData wrapper."""
    return FFData(parse_ffe_dataset(path))
