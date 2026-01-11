from . import _parser # type: ignore
from functools import lru_cache


@lru_cache(maxsize=1024)
def parse_ffe(x: str):
    return _parser.parse_ffe(x)
