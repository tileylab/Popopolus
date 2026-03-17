import os
import tempfile

import numpy as np
import pandas as pd

import popopolus.diversity_statistics.theta as theta_module


def _expected_metrics_from_sfs(sfs_row, n_chromosomes):
    """Compute expected metrics for unfolded (0..n) or folded (0..floor(n/2)) SFS."""
    if len(sfs_row) == n_chromosomes + 1:
        # Unfolded SFS: use segregating bins i=1..n-1.
        segregating_bins = sfs_row[1:n_chromosomes]
        allele_counts = np.arange(1, n_chromosomes)
    else:
        # Folded SFS: use bins i=1..floor(n/2) with same true chromosome count n.
        max_folded_bin = n_chromosomes // 2
        segregating_bins = sfs_row[1:max_folded_bin + 1]
        allele_counts = np.arange(1, max_folded_bin + 1)

    s = np.sum(segregating_bins)
    n_sites = np.sum(sfs_row)
    harmonic_indices = np.arange(1, n_chromosomes)

    a1 = np.sum(1 / harmonic_indices)
    a2 = np.sum(1 / (harmonic_indices**2))
    b1 = (n_chromosomes + 1) / (3 * (n_chromosomes - 1))
    b2 = 2 * (n_chromosomes**2 + n_chromosomes + 3) / (9 * n_chromosomes * (n_chromosomes - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n_chromosomes + 2) / (a1 * n_chromosomes) + a2 / (a1**2)
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    vd = e1 * s + e2 * s * (s - 1)

    theta_pi = (
        np.sum(allele_counts * (n_chromosomes - allele_counts) * segregating_bins)
        / theta_module.combination(n_chromosomes, 2)
    )
    theta_w = s / a1
    tajima_d = (theta_pi - theta_w) / np.sqrt(vd)

    return {
        "theta_pi": theta_pi / n_sites,
        "theta_wattersons": theta_w / n_sites,
        "tajima_D": tajima_d,
    }


def _assert_close_or_both_nan(observed, expected):
    if np.isnan(observed) and np.isnan(expected):
        return
    assert np.isclose(observed, expected)


def test_estimate_thetas_unfolded_calculation(monkeypatch):
    # One population with four diploid individuals.
    ind_map = {
        "ind1": {"population": "pop1", "ploidy": 2},
        "ind2": {"population": "pop1", "ploidy": 2},
        "ind3": {"population": "pop1", "ploidy": 2},
        "ind4": {"population": "pop1", "ploidy": 2},
    }
    tax_list = ["ind1", "ind2", "ind3", "ind4"]
    genotype_dat = np.zeros((4, 1, 4), dtype=np.int8)
    intervals = [(0, 100)]

    expected_sfs = np.array([[8, 7, 6, 5, 4, 3, 2, 1, 0]], dtype=np.uint32)

    monkeypatch.setattr(theta_module, "initialize_unfolded_sfs", lambda populations, _ind_map: expected_sfs.copy())
    monkeypatch.setattr(theta_module, "occupy_unfolded_sfs", lambda populations, _tax_list, _genotype_dat, sfs: sfs)

    with tempfile.TemporaryDirectory() as temp_dir:
        theta_df = theta_module.estimate_thetas(
            genotype_dat=genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            intervals=intervals,
            folded=False,
            output_dir=temp_dir,
        )

        assert isinstance(theta_df, pd.DataFrame)
        assert len(theta_df) == 1
        assert theta_df.loc[0, "population"] == "pop1"
        assert theta_df.loc[0, "n_individuals"] == 4
        assert theta_df.loc[0, "n_chromosomes"] == 8

        expected = _expected_metrics_from_sfs(expected_sfs[0], n_chromosomes=8)
        _assert_close_or_both_nan(theta_df.loc[0, "theta_pi"], expected["theta_pi"])
        _assert_close_or_both_nan(theta_df.loc[0, "theta_wattersons"], expected["theta_wattersons"])
        _assert_close_or_both_nan(theta_df.loc[0, "tajima_D"], expected["tajima_D"])

        output_file = os.path.join(temp_dir, "theta.csv")
        assert os.path.exists(output_file)


def test_estimate_thetas_folded_calculation(monkeypatch):
    # One population with four diploid individuals.
    ind_map = {
        "ind1": {"population": "pop1", "ploidy": 2},
        "ind2": {"population": "pop1", "ploidy": 2},
        "ind3": {"population": "pop1", "ploidy": 2},
        "ind4": {"population": "pop1", "ploidy": 2},
    }
    tax_list = ["ind1", "ind2", "ind3", "ind4"]
    genotype_dat = np.zeros((4, 1, 4), dtype=np.int8)
    intervals = [(0, 100)]

    expected_sfs = np.array([[10, 7, 5, 3, 1]], dtype=np.uint32)

    monkeypatch.setattr(theta_module, "initialize_folded_sfs", lambda populations, _ind_map: expected_sfs.copy())
    monkeypatch.setattr(theta_module, "occupy_folded_sfs", lambda populations, _tax_list, _genotype_dat, sfs: sfs)

    with tempfile.TemporaryDirectory() as temp_dir:
        theta_df = theta_module.estimate_thetas(
            genotype_dat=genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            intervals=intervals,
            folded=True,
            output_dir=temp_dir,
        )

        assert isinstance(theta_df, pd.DataFrame)
        assert len(theta_df) == 1
        assert theta_df.loc[0, "population"] == "pop1"
        assert theta_df.loc[0, "n_individuals"] == 4
        assert theta_df.loc[0, "n_chromosomes"] == 8

        expected = _expected_metrics_from_sfs(expected_sfs[0], n_chromosomes=8)
        _assert_close_or_both_nan(theta_df.loc[0, "theta_pi"], expected["theta_pi"])
        _assert_close_or_both_nan(theta_df.loc[0, "theta_wattersons"], expected["theta_wattersons"])
        _assert_close_or_both_nan(theta_df.loc[0, "tajima_D"], expected["tajima_D"])

        output_file = os.path.join(temp_dir, "theta.csv")
        assert os.path.exists(output_file)
