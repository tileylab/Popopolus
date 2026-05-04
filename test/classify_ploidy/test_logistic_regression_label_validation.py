import pandas as pd
from ppgtk.classify_ploidy.logistic_regression import _load_label_table


def test_load_label_table_accepts_na_for_unknown_ploidy():
    sample_sheet = pd.DataFrame(
        {
            "sample_id": ["ind1", "ind2", "ind3"],
            "ploidy": ["2", "NA", "4"],
        }
    )

    label_df = _load_label_table(sample_sheet)

    assert label_df.loc["ind1", "ploidy"] == 2
    assert pd.isna(label_df.loc["ind2", "ploidy"])
    assert label_df.loc["ind3", "ploidy"] == 4


def test_load_label_table_coerces_empty_ploidy_values_to_na():
    sample_sheet = pd.DataFrame(
        {
            "sample_id": ["ind1", "ind2"],
            "ploidy": ["2", ""],
        }
    )

    label_df = _load_label_table(sample_sheet)
    assert label_df.loc["ind1", "ploidy"] == 2
    assert pd.isna(label_df.loc["ind2", "ploidy"])


def test_load_label_table_coerces_zero_ploidy_values_to_na():
    sample_sheet = pd.DataFrame(
        {
            "sample_id": ["ind1", "ind2"],
            "ploidy": [2, 0],
        }
    )

    label_df = _load_label_table(sample_sheet)
    assert label_df.loc["ind1", "ploidy"] == 2
    assert pd.isna(label_df.loc["ind2", "ploidy"])


def test_load_label_table_coerces_other_invalid_ploidy_values_to_na():
    sample_sheet = pd.DataFrame(
        {
            "sample_id": ["ind1", "ind2", "ind3", "ind4"],
            "ploidy": ["4", "-2", "2.5", "abc"],
        }
    )

    label_df = _load_label_table(sample_sheet)
    assert label_df.loc["ind1", "ploidy"] == 4
    assert pd.isna(label_df.loc["ind2", "ploidy"])
    assert pd.isna(label_df.loc["ind3", "ploidy"])
    assert pd.isna(label_df.loc["ind4", "ploidy"])
