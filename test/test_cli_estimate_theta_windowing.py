import numpy as np
import pandas as pd
from click.testing import CliRunner

import popopolus_cli
import popopolus.utils as utils_mod
from popopolus.calculate_frequencies import calculate_frequencies as calc_freq_mod
from popopolus.diversity_statistics import theta as theta_mod


def test_estimate_theta_windowing_integration(tmp_path, monkeypatch):
    runner = CliRunner()

    def fake_map_individuals(_sample_sheet):
        return {
            "ind1": {"population": "pop1", "ploidy": 2},
            "ind2": {"population": "pop1", "ploidy": 2},
        }

    def fake_get_vcf_dimensions(_vcf_file, _pass_flag, _ind_map):
        return 3, 2

    def fake_get_ind_genotypes(
        _n_sites,
        _n_tax,
        _ind_map,
        _vcf_file,
        _min_depth,
        _min_count,
        _min_qual,
        _pass_flag,
        _output_dir,
        return_site_data=False,
    ):
        tax_list = ["ind1", "ind2"]
        genotype_dat = np.zeros((4, 3, 2), dtype=np.int8)
        site_df = pd.DataFrame(
            {
                "site_index": [0, 1, 2],
                "chromosome": ["chr1", "chr1", "chr1"],
                "chromosome_id": [0, 0, 0],
                "position": [10, 50, 160],
            }
        )
        if return_site_data:
            return tax_list, genotype_dat, site_df
        return tax_list, genotype_dat

    def fake_estimate_thetas(genotype_dat, _tax_list, _ind_map, _interval, _folded, _output_dir):
        n_sites = int(genotype_dat.shape[1])
        return pd.DataFrame(
            [
                {
                    "population": "pop1",
                    "n_individuals": 2,
                    "n_chromosomes": 4,
                    "theta_wattersons": float(n_sites),
                    "theta_pi": float(n_sites),
                    "tajima_D": 0.0,
                }
            ]
        )

    monkeypatch.setattr(utils_mod, "map_individuals", fake_map_individuals)
    monkeypatch.setattr(utils_mod, "get_vcf_dimensions", fake_get_vcf_dimensions)
    monkeypatch.setattr(calc_freq_mod, "get_ind_genotypes", fake_get_ind_genotypes)
    monkeypatch.setattr(theta_mod, "estimate_thetas", fake_estimate_thetas)

    out_dir = tmp_path / "theta_out"
    result = runner.invoke(
        popopolus_cli.cli,
        [
            "estimate-theta",
            "samples.csv",
            "-v",
            "input.vcf",
            "-l",
            "100:100",
            "-o",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0

    theta_df = pd.read_csv(out_dir / "theta.csv")
    assert "window_id" in theta_df.columns
    assert "chromosome" in theta_df.columns
    assert "start" in theta_df.columns
    assert "end" in theta_df.columns
    assert "n_sites_window" in theta_df.columns
    assert theta_df.shape[0] == 2
    assert theta_df["n_sites_window"].tolist() == [2, 1]

    windows_df = pd.read_csv(out_dir / "windows.csv")
    assert windows_df.shape[0] == 2
