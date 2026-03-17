# Popopolus
[![CI](https://github.com/gtiley/popopolus/actions/workflows/ci-install.yml/badge.svg?branch=main)](https://github.com/gtiley/popopolus/actions/workflows/ci-install.yml)

Python package for polyploid population genomics analyses and data exploration

## Disclaimer
This package is in development and not intended for use.

## Installation

### Conda Install
```python
conda env create -f environment.yml
conda activate popopolus
```
The *environment.yml* file was created from the popopolus development environment with `conda env export --from-history > environment.yml`.

### Pip Install
A pip installation can be done system-wide or within a new conda/venv environment. The recommended install path is to install from package metadata so dependency resolution stays platform-aware. This install path is tested on Python 3.11 and 3.12 across Ubuntu, macOS, and Windows.

```python
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install .
```

If you need a fully pinned developer environment, `requirements.txt` is still available, but pinned transitive dependencies can be less portable across operating systems.
