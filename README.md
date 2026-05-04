# ppgtk
[![CI](https://github.com/gtiley/ppgtk/actions/workflows/ci-install.yml/badge.svg?branch=main)](https://github.com/gtiley/ppgtk/actions/workflows/ci-install.yml)

Python package for polyploid population genomics analyses and data exploration

## Disclaimer
This package is in development and not intended for use.

## Installation

### Conda Install
```python
conda env create -f environment.yml
conda activate ppgtk
```
The *environment.yml* file was created from the ppgtk development environment with `conda env export --from-history > environment.yml`.

### Pip Install
A pip installation can be done system-wide or within a new conda/venv environment. The recommended install path is to install from package metadata so dependency resolution stays platform-aware. This install path is tested on Python 3.11 and 3.12 across Ubuntu, macOS, and Windows.

```python
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install .
```

If you need a fully pinned developer environment, `requirements.txt` is still available, but pinned transitive dependencies can be less portable across operating systems.

## Classifier accuracy metrics

The ploidy classifier (logistic regression) prints and writes several performance artifacts when run via the `logistic_regression` entrypoint. These are produced by the `cross_validate_logistic_regression` and `evaluate_logistic_regression_model` helpers and can be found in the output directory when provided.

- `logistic_regression_cv_accuracy.csv` — a single-row table with summary metrics:
	- `accuracy`: overall fraction of correctly classified samples
	- `balanced_accuracy`: per-class recall averaged equally across classes (useful for imbalanced label sets)
	- `macro_f1`: unweighted mean of per-class F1 scores
	- `weighted_f1`: mean of per-class F1 scores weighted by support (class prevalence)

- `logistic_regression_cv_report.csv` — the full sklearn classification report (precision / recall / f1 / support) for each class. Use this to inspect per-ploidy performance.

- `logistic_regression_cv_confusion_matrix.csv` — confusion matrix (rows = true labels, columns = predicted labels) showing raw classification counts.

- `logistic_regression_predictions.csv` — per-sample results including:
	- `known_ploidy` (when available), `predicted_ploidy`
	- probability columns named `prob_<class_label>` for class-wise predicted probabilities

Interpreting the metrics:
- Use `balanced_accuracy` and `macro_f1` when classes are imbalanced to avoid dominance by the largest class.
- `weighted_f1` is helpful when you want a single F1 that reflects the dataset's class proportions.
- The confusion matrix and per-class precision/recall help identify which ploidy levels are being confused by the model.

When adding or removing features (for example the new `mean_ab` / `median_ab` summaries), re-run cross-validation and compare these artifacts to track improvements or regressions.
