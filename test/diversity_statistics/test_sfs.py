import numpy as np

from ppgtk.diversity_statistics.sfs import initialize_folded_sfs
from ppgtk.diversity_statistics.sfs import initialize_unfolded_sfs
from ppgtk.diversity_statistics.sfs import occupy_folded_sfs
from ppgtk.diversity_statistics.sfs import occupy_unfolded_sfs
from ppgtk.utils import assign_populations


def _build_test_inputs():
    ind_map = {
        "ind1": {"population": "pop1", "ploidy": 2},
        "ind2": {"population": "pop1", "ploidy": 2},
    }
    tax_list = ["ind1", "ind2"]

    # Rows are sites and columns are individuals.
    # Site 0: [0,0] => derived count 0
    # Site 1: [1,0] => derived count 1
    # Site 2: [2,0] => derived count 2
    # Site 3: [2,2] => derived count 4
    genotype_layer = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 2],
        ],
        dtype=np.int8,
    )

    n_sites, n_individuals = genotype_layer.shape
    genotype_dat = np.zeros((4, n_sites, n_individuals), dtype=np.int16)
    genotype_dat[0] = genotype_layer

    populations = assign_populations(ind_map)
    return populations, tax_list, genotype_dat, ind_map


def test_unfolded_sfs_from_genotype_matrix():
    populations, tax_list, genotype_dat, ind_map = _build_test_inputs()

    sfs = initialize_unfolded_sfs(populations, ind_map)
    assert sfs.shape == (1, 5)

    observed = occupy_unfolded_sfs(populations, tax_list, genotype_dat, sfs)
    expected = np.array([[1, 1, 1, 0, 1]], dtype=np.uint32)
    assert np.array_equal(observed, expected)


def test_folded_sfs_from_genotype_matrix():
    populations, tax_list, genotype_dat, ind_map = _build_test_inputs()

    sfs = initialize_folded_sfs(populations, ind_map)
    assert sfs.shape == (1, 3)

    observed = occupy_folded_sfs(populations, tax_list, genotype_dat, sfs)
    expected = np.array([[2, 1, 1]], dtype=np.uint32)
    assert np.array_equal(observed, expected)
