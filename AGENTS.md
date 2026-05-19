# Repository Guidelines

## Project Structure & Module Organization

This repository packages an FFE far-field parser implemented with Python and a C++ pybind11 extension. Runtime Python code lives in `src/ffe/`: `parser.py` wraps the compiled parser, `data/` contains data containers, and `utils/ffe2xr.py` converts parsed FFE objects into `xarray` datasets. The C++ parser source is `cpp/parser.cpp` and is exposed as `ffe._parser` by `setup.py`. `example/ffe_parse_test.py` is an executable usage example. Root files such as `pyproject.toml`, `setup.py`, and `uv.lock` define packaging and dependency state.

## Build, Test, and Development Commands

- `uv sync`: install the locked Python environment.
- `uv pip install -e .`: install the package in editable mode and build the pybind11 extension.
- `python example/ffe_parse_test.py`: run the sample parser workflow against `.data/test.ffe` when that local data file is available.
- `python -m build`: build source and wheel distributions, if the `build` package is installed.

The extension build requires a working C++ compiler plus Python headers compatible with the active interpreter.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and keep imports grouped as standard library, third-party, then local modules. Follow the existing 4-space indentation style. Use `snake_case` for functions, modules, and variables, and `PascalCase` for classes such as `FFEToXarray`. Keep public parser APIs small and prefer typed, path-like inputs where practical. In C++, keep pybind11 bindings close to the structs/functions they expose and use clear ownership boundaries between parsing, data structures, and Python conversion.

## Testing Guidelines

There is no formal test suite committed yet. Treat `example/ffe_parse_test.py` as a smoke test, but add proper tests under a future `tests/` directory for new behavior. Name Python tests `test_*.py`, keep fixture `.ffe` files small, and cover both parser output shape and xarray coordinate/variable conversion. When adding tests, prefer `pytest` and document the exact command, for example `uv run pytest`.

## Commit & Pull Request Guidelines

Recent history uses short Chinese commit summaries such as `修复Merge错误` and `添加FFE obj to Xarray转换类`. Keep commits concise and action-oriented; one logical change per commit is preferred. Pull requests should describe the parser or packaging behavior changed, list verification commands run, link related issues, and include sample input/output notes when `.ffe` parsing behavior changes.

## Security & Configuration Tips

Do not commit local `.data/` files, large simulation outputs, compiler artifacts, or machine-specific environment files. Keep dependency changes synchronized between `pyproject.toml` and `uv.lock`.
