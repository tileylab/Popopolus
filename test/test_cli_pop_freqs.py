import numpy as np
import pandas as pd
from click.testing import CliRunner

import ppgtk_cli
import ppgtk.utils as utils_mod
from ppgtk.calculate_frequencies import calculate_frequencies as calc_freq_mod


def test_cli_population_frequencies_integration(tmp_path, monkeypatch):
    runner = CliRunner()

    def fake_map_individuals(_sample_sheet):
        return {
            'ind1': {'population': 'pop1', 'ploidy': 2},
            'ind2': {'population': 'pop1', 'ploidy': 2},
            'ind3': {'population': 'pop2', 'ploidy': 2},
        }

    def fake_get_vcf_dimensions(_vcf_file, _pass_flag, _ind_map):
        return 3, 3

    def fake_get_ind_genotypes(
        _n_sites, _n_tax, _ind_map, _vcf_file,
        _min_depth, _min_count, _min_qual, _pass_flag,
        _output_dir, return_site_data=False,
    ):
        tax_list = ['ind1', 'ind2', 'ind3']
        dosage = np.array([[0, 2, 1], [1, 1, 0], [2, 0, 1]], dtype=np.int8)
        depth = np.full((3, 3), 20, dtype=np.uint16)
        quality = np.full((3, 3), 50, dtype=np.uint8)
        passing = np.ones((3, 3), dtype=np.bool_)
        genotype_dat = np.array([dosage, depth, quality, passing])
        site_df = pd.DataFrame({
            'site_index': [0, 1, 2],
            'chromosome': ['chr1', 'chr1', 'chr2'],
            'chromosome_id': [0, 0, 1],
            'position': [100, 200, 50],
            'ref_allele': ['A', 'C', 'G'],
            'alt_allele': ['T', 'A', 'C'],
        })
        if return_site_data:
            return tax_list, genotype_dat, site_df
        return tax_list, genotype_dat

    monkeypatch.setattr(utils_mod, 'map_individuals', fake_map_individuals)
    monkeypatch.setattr(utils_mod, 'get_vcf_dimensions', fake_get_vcf_dimensions)
    monkeypatch.setattr(calc_freq_mod, 'get_ind_genotypes', fake_get_ind_genotypes)

    out_dir = tmp_path / 'freq_out'
    result = runner.invoke(
        ppgtk_cli.cli,
        [
            'population-frequencies',
            'samples.csv',
            '-v', 'input.vcf',
            '-o', str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    freq_df = pd.read_csv(out_dir / 'population_frequencies.csv', index_col=0)
    assert freq_df.shape[0] == 2  # 2 populations
    assert freq_df.shape[1] == 3  # 3 loci
    assert 'pop1' in freq_df.index
    assert 'pop2' in freq_df.index
