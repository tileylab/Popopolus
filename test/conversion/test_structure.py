import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from popopolus.conversion.structure import build_structure_matrix, write_structure_file


def _make_site_df(ref_alleles, alt_alleles, chromosomes=None, positions=None):
    n = len(ref_alleles)
    if chromosomes is None:
        chromosomes = ['chr1'] * n
    if positions is None:
        positions = list(range(100, 100 + n * 100, 100))
    return pd.DataFrame({
        'site_index': list(range(n)),
        'chromosome': chromosomes,
        'chromosome_id': [0] * n,
        'position': positions,
        'ref_allele': ref_alleles,
        'alt_allele': alt_alleles,
    })


def test_build_structure_matrix_diploid():
    # 2 diploid individuals in 2 populations, 3 sites
    tax_list = ['ind1', 'ind2']
    ind_map = {
        'ind1': {'population': 'popB', 'ploidy': 2},
        'ind2': {'population': 'popA', 'ploidy': 2},
    }
    # Layer 0: dosage. ind1: [0, 1, -1], ind2: [2, 1, 0]
    dosage = np.array([[0, 2], [1, 1], [-1, 0]], dtype=np.int8)
    depth = np.full((3, 2), 20, dtype=np.uint16)
    quality = np.full((3, 2), 50, dtype=np.uint8)
    passing = np.ones((3, 2), dtype=np.bool_)
    genotype_dat = np.array([dosage, depth, quality, passing])

    site_df = _make_site_df(['A', 'C', 'G'], ['T', 'A', 'C'])

    locus_labels, data_rows = build_structure_matrix(genotype_dat, tax_list, ind_map, site_df)

    # 3 loci
    assert len(locus_labels) == 3
    assert locus_labels[0] == 'chr1_100'

    # 2 individuals * 2 ploidy = 4 rows
    assert len(data_rows) == 4

    # popA (ind2) comes first (sorted), then popB (ind1)
    assert data_rows[0][0] == 'ind2'
    assert data_rows[1][0] == 'ind2'
    assert data_rows[2][0] == 'ind1'
    assert data_rows[3][0] == 'ind1'

    # Population numbers: popA=1, popB=2
    assert data_rows[0][1] == 1  # ind2 in popA
    assert data_rows[2][1] == 2  # ind1 in popB

    # ind2 at site 0: dosage=2 (hom alt T), ref=A(1), alt=T(4)
    # allele copy 0: k=0 < dosage=2 → alt=4
    # allele copy 1: k=1 < dosage=2 → alt=4
    assert data_rows[0][2] == 4  # T
    assert data_rows[1][2] == 4  # T

    # ind1 at site 0: dosage=0 (hom ref A), ref=A(1), alt=T(4)
    # allele copy 0: k=0 < dosage=0 → False → ref=1
    # allele copy 1: k=1 < dosage=0 → False → ref=1
    assert data_rows[2][2] == 1  # A
    assert data_rows[3][2] == 1  # A

    # ind1 at site 1: dosage=1 (het), ref=C(2), alt=A(1)
    # allele copy 0: k=0 < 1 → alt=1
    # allele copy 1: k=1 < 1 → False → ref=2
    assert data_rows[2][3] == 1  # A (alt)
    assert data_rows[3][3] == 2  # C (ref)

    # ind1 at site 2: dosage=-1 (missing) → -9
    assert data_rows[2][4] == -9
    assert data_rows[3][4] == -9


def test_build_structure_matrix_polyploid():
    # 1 tetraploid individual, 2 sites
    tax_list = ['ind1']
    ind_map = {'ind1': {'population': 'pop1', 'ploidy': 4}}

    # dosage 3 at site 0, dosage 0 at site 1
    dosage = np.array([[3], [0]], dtype=np.int8)
    depth = np.full((2, 1), 20, dtype=np.uint16)
    quality = np.full((2, 1), 50, dtype=np.uint8)
    passing = np.ones((2, 1), dtype=np.bool_)
    genotype_dat = np.array([dosage, depth, quality, passing])

    site_df = _make_site_df(['A', 'G'], ['T', 'C'])

    locus_labels, data_rows = build_structure_matrix(genotype_dat, tax_list, ind_map, site_df)

    # 4 rows for tetraploid
    assert len(data_rows) == 4

    # Site 0: dosage=3, ref=A(1), alt=T(4)
    # Copies 0,1,2 → alt(4), copy 3 → ref(1)
    assert [row[2] for row in data_rows] == [4, 4, 4, 1]

    # Site 1: dosage=0, ref=G(3), alt=C(2)
    # All copies → ref(3)
    assert [row[3] for row in data_rows] == [3, 3, 3, 3]


def test_write_structure_file_roundtrip(tmp_path):
    locus_labels = ['chr1_100', 'chr1_200']
    data_rows = [
        ['ind1', 1, 1, 4],
        ['ind1', 1, 4, 1],
    ]
    output_path = tmp_path / 'test.str'
    write_structure_file(locus_labels, data_rows, str(output_path))

    lines = output_path.read_text().strip().split('\n')
    assert len(lines) == 3  # 1 header + 2 data rows

    # Header has locus labels
    header_parts = lines[0].split('\t')
    assert 'chr1_100' in header_parts
    assert 'chr1_200' in header_parts

    # Data rows
    row1 = lines[1].split('\t')
    assert row1[0] == 'ind1'
    assert row1[1] == '1'


def test_cli_vcf_to_structure_integration(tmp_path, monkeypatch):
    import popopolus_cli
    import popopolus.utils as utils_mod
    from popopolus.calculate_frequencies import calculate_frequencies as calc_freq_mod

    runner = CliRunner()

    def fake_map_individuals(_sample_sheet):
        return {
            'ind1': {'population': 'pop1', 'ploidy': 2},
            'ind2': {'population': 'pop2', 'ploidy': 2},
        }

    def fake_get_vcf_dimensions(_vcf_file, _pass_flag, _ind_map):
        return 3, 2

    def fake_get_ind_genotypes(
        _n_sites, _n_tax, _ind_map, _vcf_file,
        _min_depth, _min_count, _min_qual, _pass_flag,
        _output_dir, return_site_data=False,
    ):
        tax_list = ['ind1', 'ind2']
        dosage = np.array([[0, 2], [1, 1], [2, 0]], dtype=np.int8)
        depth = np.full((3, 2), 20, dtype=np.uint16)
        quality = np.full((3, 2), 50, dtype=np.uint8)
        passing = np.ones((3, 2), dtype=np.bool_)
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

    output_file = tmp_path / 'output.str'
    result = runner.invoke(
        popopolus_cli.cli,
        [
            'vcf-to-structure',
            'samples.csv',
            '-v', 'input.vcf',
            '-o', str(output_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_file.exists()

    lines = output_file.read_text().strip().split('\n')
    # Header + 2 individuals * 2 ploidy = 5 lines
    assert len(lines) == 5

    # Verify locus labels in header
    header = lines[0]
    assert 'chr1_100' in header
    assert 'chr2_50' in header
