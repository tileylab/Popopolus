import numpy as np

from ppgtk.calculate_frequencies.calculate_frequencies import get_ind_genotypes
from ppgtk.utils import get_vcf_dimensions


def test_get_vcf_dimensions_respects_pass_flag_string(tmp_path):
    vcf = tmp_path / 'tiny.vcf'
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "chr1\t10\t.\tA\tG\t.\tPASS\t.\tGT:AD:DP:GQ:PL\t0/1:5,5:10:35:0,35,120\n"
        "chr1\t20\t.\tC\tT\t.\t.\t.\tGT:AD:DP:GQ:PL\t0/1:6,4:10:40:0,40,150\n",
        encoding='ascii',
    )

    ind_map = {'s1': {'population': 'pop1', 'ploidy': 2}}
    n_sites_dot, n_tax_dot = get_vcf_dimensions(str(vcf), '.', ind_map)
    n_sites_pass, n_tax_pass = get_vcf_dimensions(str(vcf), 'PASS', ind_map)

    assert n_tax_dot == 1
    assert n_tax_pass == 1
    assert n_sites_dot == 1
    assert n_sites_pass == 1


def test_get_ind_genotypes_reads_gq_by_format_key(tmp_path):
    vcf = tmp_path / 'format_order.vcf'
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "chr1\t10\t.\tA\tG\t.\tPASS\t.\tGT:AD:DP:PL:GQ\t0/1:5,6:11:0,523,900:42\n",
        encoding='ascii',
    )

    ind_map = {'s1': {'population': 'pop1', 'ploidy': 2}}
    n_sites, n_tax = get_vcf_dimensions(str(vcf), 'PASS', ind_map)
    tax_list, genotype_dat = get_ind_genotypes(
        n_sites=n_sites,
        n_tax=n_tax,
        ind_map=ind_map,
        vcf_file=str(vcf),
        min_depth=0,
        min_count=0,
        min_qual=0,
        pass_flag='PASS',
        output_dir='dummy',
    )

    assert tax_list == ['s1']
    assert genotype_dat.shape == (4, 1, 1)
    assert int(genotype_dat[0, 0, 0]) == 1
    assert int(genotype_dat[1, 0, 0]) == 11
    assert int(genotype_dat[2, 0, 0]) == 42
    assert int(genotype_dat[3, 0, 0]) == 1
    assert np.max(genotype_dat[2]) <= 99


def test_get_ind_genotypes_without_gq_does_not_misparse_quality(tmp_path):
    vcf = tmp_path / 'no_gq.vcf'
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "chr1\t10\t.\tA\tG\t.\tPASS\t.\tGT:DP:AD:RO:QR:AO:QA:GL\t0/1:20:10,10:10:300:10:300:-20,-1,-20\n",
        encoding='ascii',
    )

    ind_map = {'s1': {'population': 'pop1', 'ploidy': 2}}
    n_sites, n_tax = get_vcf_dimensions(str(vcf), 'PASS', ind_map)
    _, genotype_dat = get_ind_genotypes(
        n_sites=n_sites,
        n_tax=n_tax,
        ind_map=ind_map,
        vcf_file=str(vcf),
        min_depth=10,
        min_count=3,
        min_qual=20,
        pass_flag='PASS',
        output_dir='dummy',
    )

    # No GQ key exists, so parser should not use RO/QR/etc as a surrogate quality value.
    assert int(genotype_dat[2, 0, 0]) == 0
    assert int(genotype_dat[3, 0, 0]) == 1
