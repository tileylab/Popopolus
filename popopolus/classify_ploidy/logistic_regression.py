"""Utilities for ploidy classification with logistic regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EXPECTED_BINS: tuple[float, ...] = (
    0.20,
    0.25,
    0.33,
    0.40,
    0.50,
    0.60,
    0.66,
    0.75,
    0.80,
)
DEFAULT_HETEROZYGOSITY_COLUMN = "heterozygosity_rate"
DEFAULT_MEAN_AB_COLUMN = "mean_ab"
DEFAULT_MEDIAN_AB_COLUMN = "median_ab"
DEFAULT_LABEL_COLUMN = "ploidy"
DEFAULT_SAMPLE_ID_COLUMN = "sample_id"
DEFAULT_TOTAL_HET_COLUMN = "total_het_sites"


def _bin_column_name(bin_value: float) -> str:
    return f"bin_{str(bin_value).replace('.', '_')}"


def get_bin_columns(expected_bins: Sequence[float] = DEFAULT_EXPECTED_BINS) -> list[str]:
    """Return the canonical allele-balance bin column names."""

    return [_bin_column_name(bin_value) for bin_value in expected_bins]


def _closest_bin(value: float, bins: Sequence[float]) -> float:
    return min(bins, key=lambda bin_value: abs(value - bin_value))


def _coerce_column_list(columns: str | Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _validate_required_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required {context} columns: {missing_columns}."
        )


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.copy()
    for column in numeric_df.columns:
        try:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Feature column '{column}' must be numeric.") from exc
    return numeric_df


def _coerce_labels(labels: pd.Series | Sequence[Any], index: pd.Index) -> pd.Series:
    if isinstance(labels, pd.Series):
        label_series = labels.copy()
        if not label_series.index.equals(index):
            label_series = label_series.reindex(index)
    else:
        label_series = pd.Series(labels, index=index, name=DEFAULT_LABEL_COLUMN)
    return label_series


def _validate_no_missing_features(features: pd.DataFrame) -> None:
    missing_mask = features.isna()
    if missing_mask.any().any():
        missing_columns = features.columns[missing_mask.any()].tolist()
        raise ValueError(
            "Feature matrix contains missing values in columns "
            f"{missing_columns}. Fill or impute missing values before fitting "
            "or prediction."
        )


def _validate_labels(labels: pd.Series) -> None:
    if labels.isna().any():
        raise ValueError(
            f"Label column '{labels.name or DEFAULT_LABEL_COLUMN}' contains missing values."
        )
    if labels.nunique(dropna=True) < 2:
        raise ValueError("Logistic regression requires at least two label classes.")


def _resolve_label_order(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: Sequence[Any] | None = None,
) -> list[Any]:
    if label_order is not None:
        return list(label_order)
    return pd.Index(y_true).append(pd.Index(y_pred)).unique().tolist()


def _load_label_table(
    sample_sheet: str | Path | pd.DataFrame,
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    sample_id_column: str | None = None,
) -> pd.DataFrame:
    if isinstance(sample_sheet, pd.DataFrame):
        label_df = sample_sheet.copy()
    else:
        label_df = pd.read_csv(sample_sheet, sep=None, engine="python")

    if label_column not in label_df.columns:
        raise ValueError(
            f"Label table is missing the required '{label_column}' column."
        )

    if sample_id_column is None:
        for candidate in (DEFAULT_SAMPLE_ID_COLUMN, "individual"):
            if candidate in label_df.columns:
                sample_id_column = candidate
                break

    if sample_id_column is None:
        if label_df.index.is_unique and not isinstance(label_df.index, pd.RangeIndex):
            label_df.index = label_df.index.astype(str).str.strip()
            return label_df
        raise ValueError(
            "Could not infer the sample ID column. Provide a table with a "
            f"'{DEFAULT_SAMPLE_ID_COLUMN}' or 'individual' column."
        )

    label_df = label_df.copy()
    label_df[sample_id_column] = label_df[sample_id_column].astype(str).str.strip()
    return label_df.set_index(sample_id_column)


def extract_allele_balance_features(
    vcf_file: str | Path,
    *,
    expected_bins: Sequence[float] = DEFAULT_EXPECTED_BINS,
    sample_id_column: str = DEFAULT_SAMPLE_ID_COLUMN,
) -> pd.DataFrame:
    """Build sample-level allele-balance features directly from a VCF file."""

    try:
        from cyvcf2 import VCF
    except ImportError as exc:
        raise ImportError(
            "extract_allele_balance_features requires cyvcf2. "
            "Install cyvcf2 to derive features directly from a VCF."
        ) from exc

    bin_columns = get_bin_columns(expected_bins)
    individual_ratios: dict[str, list[float]] = {}
    heterozygous_counts: dict[str, int] = {}
    nonmissing_counts: dict[str, int] = {}

    vcf = VCF(str(vcf_file))
    samples = list(vcf.samples)
    for sample_id in samples:
        individual_ratios[sample_id] = []
        heterozygous_counts[sample_id] = 0
        nonmissing_counts[sample_id] = 0

    for variant in vcf:
        if len(variant.ALT) != 1:
            continue

        genotype_rows = variant.genotypes
        allelic_depths = variant.format("AD")

        for sample_index, sample_id in enumerate(samples):
            genotype = genotype_rows[sample_index]
            if len(genotype) < 2:
                continue

            genotype_0, genotype_1 = genotype[0], genotype[1]
            if (
                genotype_0 is None
                or genotype_1 is None
                or genotype_0 < 0
                or genotype_1 < 0
            ):
                continue

            nonmissing_counts[sample_id] += 1

            is_heterozygous = (
                (genotype_0 == 0 and genotype_1 == 1)
                or (genotype_0 == 1 and genotype_1 == 0)
            )
            if not is_heterozygous:
                continue

            heterozygous_counts[sample_id] += 1
            if allelic_depths is None:
                continue

            sample_depths = allelic_depths[sample_index]
            if len(sample_depths) < 2:
                continue

            ref_depth = sample_depths[0]
            alt_depth = sample_depths[1]
            if (
                ref_depth is None
                or alt_depth is None
                or ref_depth < 0
                or alt_depth < 0
            ):
                continue

            depth_total = ref_depth + alt_depth
            if depth_total <= 0:
                continue

            ratio = alt_depth / depth_total
            if 0.0 <= ratio <= 1.0:
                individual_ratios[sample_id].append(float(ratio))

    feature_rows = []
    for sample_id in samples:
        bin_count_row = {column: 0 for column in bin_columns}
        ratios = individual_ratios[sample_id]

        if ratios:
            binned_ratios = [_closest_bin(ratio, expected_bins) for ratio in ratios]
            for bin_value, count in pd.Series(binned_ratios).value_counts().items():
                bin_count_row[_bin_column_name(bin_value)] = int(count)

        feature_rows.append({sample_id_column: sample_id, **bin_count_row})

    feature_df = pd.DataFrame(feature_rows).set_index(sample_id_column)
    feature_df[DEFAULT_HETEROZYGOSITY_COLUMN] = [
        (
            heterozygous_counts[sample_id] / nonmissing_counts[sample_id]
            if nonmissing_counts[sample_id] > 0
            else np.nan
        )
        for sample_id in feature_df.index
    ]
    feature_df[DEFAULT_TOTAL_HET_COLUMN] = feature_df[bin_columns].sum(axis=1)

    feature_df[DEFAULT_MEAN_AB_COLUMN] = [
        float(np.mean(individual_ratios[sample_id]))
        if individual_ratios[sample_id]
        else np.nan
        for sample_id in feature_df.index
    ]
    feature_df[DEFAULT_MEDIAN_AB_COLUMN] = [
        float(np.median(individual_ratios[sample_id]))
        if individual_ratios[sample_id]
        else np.nan
        for sample_id in feature_df.index
    ]

    denominator = feature_df[DEFAULT_TOTAL_HET_COLUMN].replace(0, np.nan)
    for bin_column in bin_columns:
        feature_df[f"{bin_column}_prop"] = feature_df[bin_column] / denominator

    return feature_df


def prepare_feature_columns(
    df: pd.DataFrame,
    *,
    label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    bin_columns: Sequence[str] | None = None,
    heterozygosity_column: str | None = DEFAULT_HETEROZYGOSITY_COLUMN,
    fill_value: float | None = None,
    drop_missing_labels: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Select the feature matrix and optional labels for logistic regression."""

    prepared_df = df.copy()
    resolved_bin_columns = _coerce_column_list(bin_columns) or get_bin_columns()

    if feature_columns is not None:
        selected_columns = _coerce_column_list(feature_columns) or []
        requested_proportion_columns = [
            column for column in selected_columns if column.endswith("_prop")
        ]
        missing_requested_proportions = [
            column
            for column in requested_proportion_columns
            if column not in prepared_df.columns and column[:-5] in prepared_df.columns
        ]
        if missing_requested_proportions:
            denominator_columns = (
                resolved_bin_columns
                if all(column in prepared_df.columns for column in resolved_bin_columns)
                else [column[:-5] for column in requested_proportion_columns]
            )
            if DEFAULT_TOTAL_HET_COLUMN not in prepared_df.columns:
                prepared_df[DEFAULT_TOTAL_HET_COLUMN] = prepared_df[denominator_columns].sum(axis=1)
            denominator = prepared_df[DEFAULT_TOTAL_HET_COLUMN].replace(0, np.nan)
            for proportion_column in missing_requested_proportions:
                base_column = proportion_column[:-5]
                prepared_df[proportion_column] = prepared_df[base_column] / denominator
        _validate_required_columns(prepared_df, selected_columns, "feature")
    else:
        proportion_columns = [f"{column}_prop" for column in resolved_bin_columns]
        if all(column in prepared_df.columns for column in proportion_columns):
            selected_columns = proportion_columns
        else:
            _validate_required_columns(prepared_df, resolved_bin_columns, "bin-count feature")
            if DEFAULT_TOTAL_HET_COLUMN not in prepared_df.columns:
                prepared_df[DEFAULT_TOTAL_HET_COLUMN] = prepared_df[resolved_bin_columns].sum(axis=1)
            denominator = prepared_df[DEFAULT_TOTAL_HET_COLUMN].replace(0, np.nan)
            for bin_column, proportion_column in zip(resolved_bin_columns, proportion_columns):
                prepared_df[proportion_column] = prepared_df[bin_column] / denominator
            selected_columns = proportion_columns

        if heterozygosity_column is not None:
            _validate_required_columns(
                prepared_df,
                [heterozygosity_column],
                "heterozygosity",
            )
            selected_columns.append(heterozygosity_column)

        for ab_summary_column in (DEFAULT_MEAN_AB_COLUMN, DEFAULT_MEDIAN_AB_COLUMN):
            if ab_summary_column in prepared_df.columns:
                selected_columns.append(ab_summary_column)

    features = _coerce_numeric_frame(prepared_df[selected_columns])
    if fill_value is not None:
        features = features.fillna(fill_value)

    labels: pd.Series | None = None
    if label_column is not None:
        _validate_required_columns(prepared_df, [label_column], "label")
        labels = prepared_df[label_column].copy()
        if drop_missing_labels:
            keep_mask = labels.notna()
            features = features.loc[keep_mask]
            labels = labels.loc[keep_mask]
        elif labels.isna().any():
            missing_count = int(labels.isna().sum())
            raise ValueError(
                f"Label column '{label_column}' contains {missing_count} missing values."
            )

    return features, labels


def build_logistic_regression_pipeline(
    *,
    scale: bool = True,
    class_weight: str | dict[Any, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int | None = 0,
    solver: str = "lbfgs",
) -> Pipeline:
    """Create the scaled logistic-regression pipeline used in the notebook."""

    steps: list[tuple[str, Any]] = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "lr",
            LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
                solver=solver,
            ),
        )
    )
    return Pipeline(steps)


def train_logistic_regression_model(
    features: pd.DataFrame,
    labels: pd.Series | Sequence[Any],
    *,
    scale: bool = True,
    class_weight: str | dict[Any, float] | None = "balanced",
    max_iter: int = 1000,
    random_state: int | None = 0,
    solver: str = "lbfgs",
) -> Pipeline:
    """Fit a logistic-regression model on prepared feature columns."""

    label_series = _coerce_labels(labels, features.index)
    _validate_no_missing_features(features)
    _validate_labels(label_series)

    model = build_logistic_regression_pipeline(
        scale=scale,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
    )
    model.fit(features, label_series)
    return model


def generate_predictions(
    model: Pipeline,
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    fill_value: float | None = None,
    include_probabilities: bool = True,
) -> pd.DataFrame:
    """Generate class predictions for new samples."""

    resolved_feature_columns = _coerce_column_list(feature_columns)
    if resolved_feature_columns is None:
        resolved_feature_columns = list(getattr(model, "feature_names_in_", []))
    features, _ = prepare_feature_columns(
        df,
        feature_columns=resolved_feature_columns or None,
        fill_value=fill_value,
    )
    _validate_no_missing_features(features)

    prediction_df = pd.DataFrame(
        {"predicted_label": model.predict(features)},
        index=features.index,
    )

    if include_probabilities and hasattr(model, "predict_proba"):
        probability_matrix = model.predict_proba(features)
        probability_df = pd.DataFrame(
            probability_matrix,
            index=features.index,
            columns=[f"prob_{class_label}" for class_label in model.classes_],
        )
        prediction_df = prediction_df.join(probability_df)

    return prediction_df


def evaluate_logistic_regression_model(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series | Sequence[Any],
    *,
    label_order: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted model with classification metrics."""

    label_series = _coerce_labels(labels, features.index)
    _validate_no_missing_features(features)
    _validate_labels(label_series)

    predicted_labels = pd.Series(
        model.predict(features),
        index=features.index,
        name="predicted_label",
    )
    resolved_label_order = _resolve_label_order(label_series, predicted_labels, label_order)

    confusion = confusion_matrix(
        label_series,
        predicted_labels,
        labels=resolved_label_order,
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=resolved_label_order,
        columns=resolved_label_order,
    )

    report = classification_report(
        label_series,
        predicted_labels,
        labels=resolved_label_order,
        output_dict=True,
        zero_division=0,
    )

    accuracy_df = pd.DataFrame(
        {"accuracy": [accuracy_score(label_series, predicted_labels)],
         "balanced_accuracy": [balanced_accuracy_score(label_series, predicted_labels)],
         "macro_f1": [f1_score(label_series, predicted_labels, average="macro", zero_division=0)],
         "weighted_f1": [f1_score(label_series, predicted_labels, average="weighted", zero_division=0)]
         }
    )           

    return {
        "accuracy_df": accuracy_df,
        "classification_report": report,
        "classification_report_df": pd.DataFrame(report).transpose(),
        "confusion_matrix": confusion_df,
        "predictions": predicted_labels.to_frame(),
    }


def stratified_train_test_split(
    features: pd.DataFrame,
    labels: pd.Series | Sequence[Any],
    *,
    test_size: float = 0.2,
    random_state: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split for labeled samples."""

    label_series = _coerce_labels(labels, features.index)
    _validate_no_missing_features(features)
    _validate_labels(label_series)

    try:
        return train_test_split(
            features,
            label_series,
            test_size=test_size,
            random_state=random_state,
            stratify=label_series,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified split. Check that each class has "
            "enough labeled samples for the requested test size."
        ) from exc


def cross_validate_logistic_regression(
    features: pd.DataFrame,
    labels: pd.Series | Sequence[Any],
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 0,
    scale: bool = True,
    class_weight: str | dict[Any, float] | None = "balanced",
    max_iter: int = 1000,
    solver: str = "lbfgs",
    label_order: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run stratified cross-validation and return out-of-fold metrics."""

    label_series = _coerce_labels(labels, features.index)
    _validate_no_missing_features(features)
    _validate_labels(label_series)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    min_class_size = int(label_series.value_counts().min())
    if min_class_size < n_splits:
        raise ValueError(
            f"n_splits={n_splits} is too large for the smallest class size "
            f"({min_class_size})."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    model = build_logistic_regression_pipeline(
        scale=scale,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
    )

    predicted_labels = pd.Series(
        cross_val_predict(model, features, label_series, cv=cv),
        index=features.index,
        name="predicted_label",
    )
    resolved_label_order = _resolve_label_order(label_series, predicted_labels, label_order)

    confusion = confusion_matrix(
        label_series,
        predicted_labels,
        labels=resolved_label_order,
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=resolved_label_order,
        columns=resolved_label_order,
    )

    report = classification_report(
        label_series,
        predicted_labels,
        labels=resolved_label_order,
        output_dict=True,
        zero_division=0,
    )

    accuracy_df = pd.DataFrame(
        {"accuracy": [accuracy_score(label_series, predicted_labels)],
         "balanced_accuracy": [balanced_accuracy_score(label_series, predicted_labels)],
         "macro_f1": [f1_score(label_series, predicted_labels, average="macro", zero_division=0)],
         "weighted_f1": [f1_score(label_series, predicted_labels, average="weighted", zero_division=0)]
         }
    )

    return {
        "accuracy_df": accuracy_df,
        "classification_report": report,
        "classification_report_df": pd.DataFrame(report).transpose(),
        "confusion_matrix": confusion_df,
        "predictions": pd.DataFrame(
            {
                "true_label": label_series,
                "predicted_label": predicted_labels,
            }
        ),
        "n_splits": n_splits,
    }


def logistic_regression(
    sample_sheet: str | Path | pd.DataFrame,
    vcf_file: str | Path,
    output_dir: str | Path = "dummy",
) -> pd.DataFrame:
    """Train on labeled samples and predict ploidy from VCF-derived features."""

    feature_df = extract_allele_balance_features(vcf_file)
    label_df = _load_label_table(sample_sheet)

    merged_df = feature_df.join(label_df[[DEFAULT_LABEL_COLUMN]], how="left")
    training_df = merged_df[merged_df[DEFAULT_LABEL_COLUMN].notna()].copy()
    if training_df.empty:
        raise ValueError(
            "No labeled samples were found after merging the sample sheet with the "
            "VCF-derived feature table."
        )

    train_features, train_labels = prepare_feature_columns(
        training_df,
        label_column=DEFAULT_LABEL_COLUMN,
        fill_value=0.0,
    )
    model = train_logistic_regression_model(train_features, train_labels)

    prediction_df = generate_predictions(model, feature_df, fill_value=0.0)
    prediction_df = prediction_df.rename(columns={"predicted_label": "predicted_ploidy"})

    results_df = feature_df.join(
        label_df[[DEFAULT_LABEL_COLUMN]].rename(columns={DEFAULT_LABEL_COLUMN: "known_ploidy"}),
        how="left",
    ).join(prediction_df)

    ####
    # Evaluate performance with cross-validation on the training set
    cv_results = cross_validate_logistic_regression(train_features, train_labels)
    cv_report_df = cv_results["classification_report_df"]
    print("Cross-validation classification report:")
    print(cv_report_df)
    print("Cross-validation confusion matrix:")
    print(cv_results["confusion_matrix"])
    print("Accuracy Stats:")
    print(cv_results["accuracy_df"])
    #print(f'Accuracy: {cv_results["accuracy"]}'
    #      f'Balanced Accuracy: {cv_results["balanced_accuracy"]}'
    #      f'Macro F1: {cv_results["macro_f1"]}'
    #      f'Weighted F1: {cv_results["weighted_f1"]}')
    ####

    ####
    # Print out predictions and performance metircs to the output directory
    if output_dir != "dummy":
        output_path = Path(output_dir) / "logistic_regression_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path)
        cv_report_path = Path(output_dir) / "logistic_regression_cv_report.csv"
        cv_report_path.parent.mkdir(parents=True, exist_ok=True)
        cv_report_df.to_csv(cv_report_path)
        confusion_matrix_path = Path(output_dir) / "logistic_regression_cv_confusion_matrix.csv"
        confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results["confusion_matrix"].to_csv(confusion_matrix_path)
        accuracy_df_path = Path(output_dir) / "logistic_regression_cv_accuracy.csv"
        accuracy_df_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results["accuracy_df"].to_csv(accuracy_df_path)
    ####

    return results_df


__all__ = [
    "DEFAULT_EXPECTED_BINS",
    "DEFAULT_HETEROZYGOSITY_COLUMN",
    "DEFAULT_LABEL_COLUMN",
    "DEFAULT_MEAN_AB_COLUMN",
    "DEFAULT_MEDIAN_AB_COLUMN",
    "DEFAULT_SAMPLE_ID_COLUMN",
    "build_logistic_regression_pipeline",
    "cross_validate_logistic_regression",
    "evaluate_logistic_regression_model",
    "extract_allele_balance_features",
    "generate_predictions",
    "get_bin_columns",
    "logistic_regression",
    "prepare_feature_columns",
    "stratified_train_test_split",
    "train_logistic_regression_model",
]
