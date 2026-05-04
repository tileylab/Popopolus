# Polyploid Population Genomics Tool Kit (ppgtk)
[![CI](https://github.com/gtiley/ppgtk/actions/workflows/ci-install.yml/badge.svg?branch=main)](https://github.com/gtiley/ppgtk/actions/workflows/ci-install.yml)

Python package for polyploid population genomics analyses and data exploration

## Disclaimer
This package is in early developmental stages and a lot of functionality and specific commands are still a work in progress. The *classify-ploidy* method does work and has been vetted against several test datasets. Publication of the methods are anticipated in the near future but made available in case it is helpful for some exploratory analyses.

## Installation

### Conda Install
```python
conda env create -f environment.yml
conda activate ppgtk
```
The *environment.yml* file was created from the ppgtk development environment with `conda env export --from-history > environment.yml`.

### Pip Install
A pip installation can be done system-wide or within a new conda/venv environment. The recommended install path is to install from package metadata so dependency resolution stays platform-aware. This install path is tested on Python 3.11 and 3.12 across Ubuntu and macOS. The Windows installation is currently not working due to a limitation of a dependency for reading VCF files, but we are working towards a native parser to allow windows compatibility. If Windows is necessary for you, please submit an issue and it will happen faster.

```python
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install .
```

If you need a fully pinned developer environment, `requirements.txt` is still available, but pinned transitive dependencies can be less portable across operating systems.

## Ploidy Classification

The ploidy classification method can be implemented as a single command-line program after successful installation. To bring up the program options, you can look at the help menu:

```bash
ppgtk classify-ploidy --help
```

And an example analysis would look like this:
```bash
ppgtk classify-ploidy -v input.vcf -o results metadata.csv
```

Note that the metadata file comes at the end without a flag. This is by design. All commands in **ppgtk** require a metadata file, which has a minimal format. Only three columns are needed:
```
individual, population, ploidy
sample_1, pop_A, 2
sample_2, pop_A, 2
sample_3, pop_A, NA
...
sample_n, pop_B, 4
```
Missing values are accepted for ploidy. Missing values should only be encoded as "NA". Other numbers that are not an expected ploidy will be treated as a seperate class in the regression model. 

The population field is used when calculating population genetic statistics, but it does not matter for the *classify-ploidy* method. If you do not have *a priori* or data-driven population assignments, that is fine. You could simply duplicate the individual names or assign all individuals to a single placeholder population, such as "pop_A".

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

<!-- 
When adding or removing features (for example the new `mean_ab` / `median_ab` summaries), re-run cross-validation and compare these artifacts to track improvements or regressions.
-->