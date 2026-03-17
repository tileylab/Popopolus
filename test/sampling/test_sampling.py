import numpy as np
import pytest
import pandas as pd

from popopolus.sampling.sampling import rarefy_genotype_dataset
from popopolus.sampling.sampling import bootstrap_genotype_dataset
from popopolus.sampling.sampling import summarize_bootstrap_theta


def _build_dummy_genotype_dat(n_sites, n_taxa):
    genotype_layer = np.arange(n_sites * n_taxa, dtype=np.int8).reshape(n_sites, n_taxa)
    depth_layer = np.full((n_sites, n_taxa), 20, dtype=np.uint16)
    quality_layer = np.full((n_sites, n_taxa), 50, dtype=np.uint8)
    pass_layer = np.ones((n_sites, n_taxa), dtype=np.bool_)
    return np.array([genotype_layer, depth_layer, quality_layer, pass_layer])


def test_rarefy_genotype_dataset_with_replicates():
    tax_list = ["a1", "a2", "a3", "b1", "b2"]
    ind_map = {
        "a1": {"population": "popA", "ploidy": 2},
        "a2": {"population": "popA", "ploidy": 2},
        "a3": {"population": "popA", "ploidy": 2},
        "b1": {"population": "popB", "ploidy": 2},
        "b2": {"population": "popB", "ploidy": 2},
    }
    genotype_dat = _build_dummy_genotype_dat(n_sites=4, n_taxa=len(tax_list))

    replicate_datasets, rarefaction_df = rarefy_genotype_dataset(
        genotype_dat=genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        n_replicates=3,
        target_chromosomes=None,
        seed=17,
        require_exact=True,
    )

    assert len(replicate_datasets) == 3
    assert rarefaction_df.shape[0] == 3 * 2

    for rep in replicate_datasets:
        rep_tax_list = rep["tax_list"]
        rep_ind_map = rep["ind_map"]
        rep_genotype_dat = rep["genotype_dat"]

        assert rep_genotype_dat.shape[2] == len(rep_tax_list)
        assert len(rep_tax_list) == 4  # two diploids sampled from each population

        pop_counts = {"popA": 0, "popB": 0}
        for ind in rep_tax_list:
            pop = rep_ind_map[ind]["population"]
            pop_counts[pop] += rep_ind_map[ind]["ploidy"]

        assert pop_counts["popA"] == 4
        assert pop_counts["popB"] == 4


def test_rarefy_genotype_dataset_raises_when_exact_unreachable():
    tax_list = ["a1", "a2", "b1"]
    ind_map = {
        "a1": {"population": "popA", "ploidy": 4},
        "a2": {"population": "popA", "ploidy": 4},
        "b1": {"population": "popB", "ploidy": 5},
    }
    genotype_dat = _build_dummy_genotype_dat(n_sites=3, n_taxa=len(tax_list))

    with pytest.raises(ValueError):
        rarefy_genotype_dataset(
            genotype_dat=genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            n_replicates=1,
            target_chromosomes=None,
            seed=11,
            require_exact=True,
        )


def test_bootstrap_genotype_dataset_shape_and_count():
    genotype_dat = _build_dummy_genotype_dat(n_sites=5, n_taxa=4)
    bootstraps = bootstrap_genotype_dataset(genotype_dat, n_bootstraps=4, seed=9)

    assert len(bootstraps) == 4
    for boot in bootstraps:
        assert boot["genotype_dat"].shape == genotype_dat.shape
        assert len(boot["site_indices"]) == genotype_dat.shape[1]


def test_summarize_bootstrap_theta_returns_ci_columns():
    df = pd.DataFrame(
        [
            {
                "replicate": 1,
                "bootstrap": 1,
                "population": "pop1",
                "n_individuals": 4,
                "n_chromosomes": 8,
                "theta_wattersons": 0.10,
                "theta_pi": 0.11,
                "tajima_D": -0.5,
            },
            {
                "replicate": 1,
                "bootstrap": 2,
                "population": "pop1",
                "n_individuals": 4,
                "n_chromosomes": 8,
                "theta_wattersons": 0.20,
                "theta_pi": 0.21,
                "tajima_D": 0.3,
            },
        ]
    )

    summary = summarize_bootstrap_theta(df)

    assert summary.shape[0] == 1
    assert summary.loc[0, "n_bootstraps"] == 2
    assert "theta_wattersons_ci_lower" in summary.columns
    assert "theta_wattersons_ci_upper" in summary.columns
    assert "theta_pi_ci_lower" in summary.columns
    assert "theta_pi_ci_upper" in summary.columns
    assert "tajima_D_ci_lower" in summary.columns
    assert "tajima_D_ci_upper" in summary.columns
