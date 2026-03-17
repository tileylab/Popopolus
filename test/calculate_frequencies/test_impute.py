import numpy as np
import pandas as pd

from popopolus.calculate_frequencies.impute import average_missing
from popopolus.calculate_frequencies.impute import randomly_impute_missing
from popopolus.calculate_frequencies.impute import apply_missing_imputation
from popopolus.calculate_frequencies.impute import remove_missing


def test_average_missing_imputes_layer0_by_site_mean():
    genotype_layer = np.array(
        [
            [0, 2, -1],
            [-1, -1, 2],
            [-1, -1, -1],
        ],
        dtype=np.int16,
    )
    depth_layer = np.array(
        [
            [10, 10, 10],
            [12, 11, 13],
            [9, 9, 9],
        ],
        dtype=np.int16,
    )
    quality_layer = np.full((3, 3), 30, dtype=np.int16)
    pass_layer = np.ones((3, 3), dtype=np.int16)

    genotype_dat = np.array([genotype_layer, depth_layer, quality_layer, pass_layer])

    imputed = average_missing(genotype_dat)

    expected_layer0 = np.array(
        [
            [0, 2, 1],
            [2, 2, 2],
            [0, 0, 0],
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(imputed[0], expected_layer0)
    np.testing.assert_array_equal(imputed[1], depth_layer)
    np.testing.assert_array_equal(imputed[2], quality_layer)
    np.testing.assert_array_equal(imputed[3], pass_layer)
    # Default behavior should return a copy.
    np.testing.assert_array_equal(genotype_dat[0], genotype_layer)


def test_average_missing_in_place_updates_original():
    genotype_dat = np.array(
        [
            np.array([[0, -1]], dtype=np.int8),
            np.array([[10, 10]], dtype=np.int8),
            np.array([[30, 30]], dtype=np.int8),
            np.array([[1, 1]], dtype=np.int8),
        ]
    )

    out = average_missing(genotype_dat, in_place=True)
    assert out is genotype_dat
    np.testing.assert_array_equal(genotype_dat[0], np.array([[0, 0]], dtype=np.int8))


def test_average_missing_within_ploidy_mixed_ploidy():
    genotype_layer = np.array(
        [
            [0, -1, 4, -1],
            [2, -1, 2, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    tax_list = ["d1", "d2", "t1", "t2"]
    ind_map = {
        "d1": {"ploidy": 2},
        "d2": {"ploidy": 2},
        "t1": {"ploidy": 4},
        "t2": {"ploidy": 4},
    }

    imputed = average_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy="within_ploidy",
    )

    expected_layer0 = np.array(
        [
            [0, 0, 4, 4],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(imputed[0], expected_layer0)


def test_average_missing_scaled_mixed_ploidy():
    genotype_layer = np.array(
        [
            [1, -1, -1, -1],
            [-1, -1, 4, -1],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    tax_list = ["d1", "d2", "t1", "t2"]
    ind_map = {
        "d1": {"ploidy": 2},
        "d2": {"ploidy": 2},
        "t1": {"ploidy": 4},
        "t2": {"ploidy": 4},
    }

    imputed = average_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy="scaled",
    )

    expected_layer0 = np.array(
        [
            [1, 1, 2, 2],
            [2, 2, 4, 4],
        ],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(imputed[0], expected_layer0)


def test_randomly_impute_missing_within_ploidy_reproducible_and_bounded():
    genotype_layer = np.array(
        [
            [0, -1, 4, -1],
            [-1, -1, 3, 4],
            [-1, -1, -1, -1],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    tax_list = ["d1", "d2", "t1", "t2"]
    ind_map = {
        "d1": {"ploidy": 2},
        "d2": {"ploidy": 2},
        "t1": {"ploidy": 4},
        "t2": {"ploidy": 4},
    }

    out1 = randomly_impute_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy="within_ploidy",
        seed=123,
    )
    out2 = randomly_impute_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy="within_ploidy",
        seed=123,
    )

    np.testing.assert_array_equal(out1, out2)
    assert np.all(out1[0][:, :2] <= 2)
    assert np.all(out1[0][:, 2:] <= 4)
    assert np.all(out1[0] >= 0)


def test_randomly_impute_missing_scaled_respects_ploidy():
    genotype_layer = np.array(
        [
            [2, -1, 4, -1],
            [-1, -1, 4, 4],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    tax_list = ["d1", "d2", "t1", "t2"]
    ind_map = {
        "d1": {"ploidy": 2},
        "d2": {"ploidy": 2},
        "t1": {"ploidy": 4},
        "t2": {"ploidy": 4},
    }

    out = randomly_impute_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy="scaled",
        seed=42,
    )

    assert np.all(out[0][:, :2] <= 2)
    assert np.all(out[0][:, 2:] <= 4)
    assert np.all(out[0] >= 0)


def test_randomly_impute_missing_site_weighted_draws_observed_states():
    genotype_layer = np.array([[0, 0, 2, -1, -1]], dtype=np.int16)
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    out = randomly_impute_missing(genotype_dat, strategy="site_weighted", seed=7)
    imputed_vals = out[0][0, 3:]
    assert set(imputed_vals.tolist()).issubset({0, 2})


def test_apply_missing_imputation_dispatch_skip_and_mean():
    genotype_dat = np.array(
        [
            np.array([[0, -1]], dtype=np.int8),
            np.array([[10, 10]], dtype=np.int8),
            np.array([[30, 30]], dtype=np.int8),
            np.array([[1, 1]], dtype=np.int8),
        ]
    )

    out_skip = apply_missing_imputation(genotype_dat, method="skip")
    assert out_skip is genotype_dat

    try:
        apply_missing_imputation(genotype_dat, method="drop")
        assert False
    except ValueError:
        assert True

    out_mean = apply_missing_imputation(genotype_dat, method="mean")
    np.testing.assert_array_equal(out_mean[0], np.array([[0, 0]], dtype=np.int8))


def test_apply_missing_imputation_dispatch_unsupported():
    genotype_dat = np.array(
        [
            np.array([[0, -1]], dtype=np.int8),
            np.array([[10, 10]], dtype=np.int8),
            np.array([[30, 30]], dtype=np.int8),
            np.array([[1, 1]], dtype=np.int8),
        ]
    )

    try:
        apply_missing_imputation(genotype_dat, method="nope")
        assert False
    except ValueError:
        assert True


def test_remove_missing_filters_sites_and_updates_site_df():
    genotype_layer = np.array(
        [
            [0, 1],
            [1, -1],
            [2, 2],
            [-1, -1],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    site_df = pd.DataFrame(
        {
            "site_index": [0, 1, 2, 3],
            "chromosome": ["chr1", "chr1", "chr1", "chr2"],
            "chromosome_id": [0, 0, 0, 1],
            "position": [10, 20, 30, 40],
        }
    )

    filtered_dat, filtered_site_df, removed = remove_missing(
        genotype_dat,
        site_df=site_df,
        drop_if="any",
        return_removed=True,
    )

    np.testing.assert_array_equal(filtered_dat[0], np.array([[0, 1], [2, 2]], dtype=np.int16))
    assert filtered_site_df["position"].tolist() == [10, 30]
    assert filtered_site_df["site_index"].tolist() == [0, 1]
    np.testing.assert_array_equal(removed, np.array([1, 3], dtype=np.int64))


def test_remove_missing_drop_all_keeps_partially_observed_sites():
    genotype_layer = np.array(
        [
            [0, -1],
            [-1, -1],
            [2, 1],
        ],
        dtype=np.int16,
    )
    aux = np.zeros_like(genotype_layer)
    genotype_dat = np.array([genotype_layer, aux, aux, aux])

    filtered_dat = remove_missing(genotype_dat, drop_if="all")
    np.testing.assert_array_equal(filtered_dat[0], np.array([[0, -1], [2, 1]], dtype=np.int16))
