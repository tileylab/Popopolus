from click.testing import CliRunner

import popopolus_cli
import popopolus.utils as utils_mod
from popopolus.calculate_frequencies import calculate_frequencies as calc_freq_mod


def test_individual_genotypes_cli_integration(monkeypatch):
    runner = CliRunner()
    calls = {"get_ind_genotypes": 0}

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
        calls["get_ind_genotypes"] += 1
        assert return_site_data is False
        return ["ind1", "ind2"], None

    monkeypatch.setattr(utils_mod, "map_individuals", fake_map_individuals)
    monkeypatch.setattr(utils_mod, "get_vcf_dimensions", fake_get_vcf_dimensions)
    monkeypatch.setattr(calc_freq_mod, "get_ind_genotypes", fake_get_ind_genotypes)

    result = runner.invoke(
        popopolus_cli.cli,
        [
            "individual-genotypes",
            "samples.csv",
            "-v",
            "input.vcf",
        ],
    )

    assert result.exit_code == 0
    assert calls["get_ind_genotypes"] == 1
