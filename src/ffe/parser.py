from functools import lru_cache
from os import PathLike
from typing import Iterable, Union

from . import _parser  # type: ignore
from .data import FFData
from .utils import FFEToXarray, combine_ffe_datasets

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


@lru_cache(maxsize=1024)
def _parse_ffe_grid(path: str):
    headers, frequencies, axis1, axis2, data = _parser.parse_ffe_grid(path)
    return tuple(headers), frequencies, axis1, axis2, data


def parse_ffe(path: PathInput):
    """Parse an FFE file and return the low-level C++ FFEFile object."""
    return _parse_ffe_raw(_as_path_str(path))


def parse_ffe_array(path: PathInput):
    """Parse an FFE file into ``(headers, frequencies, data)``.

    ``data`` has shape ``(Frequency, SpatialPoint, Column)`` and crosses the
    C++/Python boundary once, which is faster than collecting each section.
    """
    return _parse_ffe_array(_as_path_str(path))


def parse_ffe_grid(path: PathInput):
    """Parse an FFE file into grid-shaped arrays.

    ``data`` has shape ``(Frequency, Axis1, Axis2, Column)`` and is ready for
    direct xarray construction.
    """
    return _parse_ffe_grid(_as_path_str(path))


def parse_ffe_dataset(path: PathInput):
    """Parse an FFE file and return an xarray Dataset."""
    return FFEToXarray.from_file(path).convert()


def parse_ffe_datasets(paths: Iterable[PathInput]):
    """Parse multiple FFE files and concatenate them along Frequency.

    Non-frequency dimensions, coordinates, and data variables must match. The
    resulting Frequency order follows the input path order.
    """
    return combine_ffe_datasets(parse_ffe_dataset(path) for path in paths)


def parse(path: PathInput | Iterable[PathInput]) -> FFData:
    """Parse one or more FFE files and return the high-level FFData wrapper."""
    if isinstance(path, (str, PathLike)):
        return FFData(parse_ffe_dataset(path))

    return FFData(parse_ffe_datasets(path))
