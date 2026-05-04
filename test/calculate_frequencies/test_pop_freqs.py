import numpy as np
import pandas as pd
import pytest

from ppgtk.calculate_frequencies.calculate_frequencies import get_pop_freqs


def _make_genotype_dat(dosage, passing=None):
    """Build a 4-layer genotype_dat from a dosage matrix and optional pass-filter matrix."""
    n_sites, n_taxa = dosage.shape
    depth = np.full((n_sites, n_taxa), 20, dtype=np.uint16)
    quality = np.full((n_sites, n_taxa), 50, dtype=np.uint8)
    if passing is None:
        passing = np.ones((n_sites, n_taxa), dtype=np.bool_)
    return np.array([dosage, depth, quality, passing])


def _make_site_df(n_sites):
    return pd.DataFrame({
        'site_index': list(range(n_sites)),
        'chromosome': ['chr1'] * n_sites,
        'chromosome_id': [0] * n_sites,
        'position': list(range(100, 100 + n_sites * 100, 100)),
    })


def test_pop_freqs_diploid():
    """Two diploid populations, 3 sites — verify frequencies match manual calculation."""
    tax_list = ['a1', 'a2', 'b1', 'b2']
    ind_map = {
        'a1': {'population': 'popA', 'ploidy': 2},
        'a2': {'population': 'popA', 'ploidy': 2},
        'b1': {'population': 'popB', 'ploidy': 2},
        'b2': {'population': 'popB', 'ploidy': 2},
    }
    # Dosages:       a1  a2  b1  b2
    # site 0:         0   2   1   1
    # site 1:         1   1   0   2
    # site 2:         2   2   0   0
    dosage = np.array([
        [0, 2, 1, 1],
        [1, 1, 0, 2],
        [2, 2, 0, 0],
    ], dtype=np.int8)
    genotype_dat = _make_genotype_dat(dosage)
    site_df = _make_site_df(3)

    freq_df = get_pop_freqs(genotype_dat, tax_list, ind_map, site_df)

    assert freq_df.shape == (2, 3)  # 2 populations, 3 sites
    assert list(freq_df.index) == ['popA', 'popB']  # sorted

    # popA site 0: (0 + 2) / (2 + 2) = 0.5
    assert freq_df.iloc[0, 0] == pytest.approx(0.5)
    # popA site 1: (1 + 1) / 4 = 0.5
    assert freq_df.iloc[0, 1] == pytest.approx(0.5)
    # popA site 2: (2 + 2) / 4 = 1.0
    assert freq_df.iloc[0, 2] == pytest.approx(1.0)
    # popB site 0: (1 + 1) / 4 = 0.5
    assert freq_df.iloc[1, 0] == pytest.approx(0.5)
    # popB site 1: (0 + 2) / 4 = 0.5
    assert freq_df.iloc[1, 1] == pytest.approx(0.5)
    # popB site 2: (0 + 0) / 4 = 0.0
    assert freq_df.iloc[1, 2] == pytest.approx(0.0)


def test_pop_freqs_mixed_ploidy():
    """Mix of diploid and tetraploid — denominator uses sum of ploidies."""
    tax_list = ['a1', 'a2']
    ind_map = {
        'a1': {'population': 'pop1', 'ploidy': 2},
        'a2': {'population': 'pop1', 'ploidy': 4},
    }
    # site 0: a1=1, a2=3 → freq = (1 + 3) / (2 + 4) = 4/6
    dosage = np.array([[1, 3]], dtype=np.int8)
    genotype_dat = _make_genotype_dat(dosage)
    site_df = _make_site_df(1)

    freq_df = get_pop_freqs(genotype_dat, tax_list, ind_map, site_df)

    assert freq_df.shape == (1, 1)
    assert freq_df.iloc[0, 0] == pytest.approx(4.0 / 6.0)


def test_pop_freqs_missing_data():
    """Sites with pass_filter=0 should be excluded from frequency calculation."""
    tax_list = ['a1', 'a2']
    ind_map = {
        'a1': {'population': 'pop1', 'ploidy': 2},
        'a2': {'population': 'pop1', 'ploidy': 2},
    }
    # site 0: a1=2 (passing), a2=0 (not passing)
    # freq should be 2/2 = 1.0 (only a1 counted)
    dosage = np.array([[2, 0]], dtype=np.int8)
    passing = np.array([[1, 0]], dtype=np.bool_)
    genotype_dat = _make_genotype_dat(dosage, passing)
    site_df = _make_site_df(1)

    freq_df = get_pop_freqs(genotype_dat, tax_list, ind_map, site_df)

    assert freq_df.iloc[0, 0] == pytest.approx(1.0)


def test_pop_freqs_all_missing_returns_nan():
    """If no individuals pass filters at a site, frequency should be NaN."""
    tax_list = ['a1']
    ind_map = {'a1': {'population': 'pop1', 'ploidy': 2}}
    dosage = np.array([[1]], dtype=np.int8)
    passing = np.array([[0]], dtype=np.bool_)
    genotype_dat = _make_genotype_dat(dosage, passing)
    site_df = _make_site_df(1)

    freq_df = get_pop_freqs(genotype_dat, tax_list, ind_map, site_df)

    assert np.isnan(freq_df.iloc[0, 0])
