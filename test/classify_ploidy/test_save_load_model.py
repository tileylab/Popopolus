import json

import numpy as np
import pandas as pd
import pytest

from popopolus.classify_ploidy.logistic_regression import (
    build_logistic_regression_pipeline,
    generate_predictions,
    load_model,
    save_model,
    train_logistic_regression_model,
)


@pytest.fixture()
def trained_model_and_features():
    """Return a small trained pipeline together with its training data."""
    rng = np.random.default_rng(42)
    n_samples = 40
    features = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n_samples),
            "feat_b": rng.normal(size=n_samples),
        },
        index=[f"sample_{i}" for i in range(n_samples)],
    )
    labels = pd.Series(
        [2] * 20 + [4] * 20,
        index=features.index,
        name="ploidy",
    )
    model = train_logistic_regression_model(features, labels)
    return model, features, labels


def test_save_creates_joblib_and_meta(tmp_path, trained_model_and_features):
    model, features, _ = trained_model_and_features
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path, feature_columns=list(features.columns))

    assert model_path.exists()
    meta_path = model_path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["feature_columns"] == ["feat_a", "feat_b"]


def test_load_restores_identical_predictions(tmp_path, trained_model_and_features):
    model, features, _ = trained_model_and_features
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path, feature_columns=list(features.columns))

    loaded_model, loaded_cols = load_model(model_path)

    assert loaded_cols == ["feat_a", "feat_b"]

    original_preds = model.predict(features)
    loaded_preds = loaded_model.predict(features)
    np.testing.assert_array_equal(original_preds, loaded_preds)


def test_load_without_meta_returns_none_columns(tmp_path, trained_model_and_features):
    model, _, _ = trained_model_and_features
    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)

    # Remove sidecar to simulate missing metadata
    model_path.with_suffix(".meta.json").unlink()

    loaded_model, loaded_cols = load_model(model_path)
    assert loaded_cols is None
    assert loaded_model is not None
