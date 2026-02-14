# Popopolus
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
A *requirements.txt* file is provided for easy pip install of dependencies. This can be done system-wide or within a new conda environment. The pip installation is tested on Python versions 3.11 and 3.12 across ubuntu, macos, and windows.

```python
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

The *requirements.txt* file was created from within the environment with `pip list --format=freeze > requirements.txt`.
