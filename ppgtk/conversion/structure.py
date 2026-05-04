import numpy as np

from ppgtk.utils import assign_populations


NUCLEOTIDE_CODE = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
MISSING_CODE = -9


def build_structure_matrix(genotype_dat, tax_list, ind_map, site_df):
    """
    Convert a genotype dosage array into STRUCTURE-format rows.

    Parameters:
        genotype_dat (np.ndarray): shape (4, n_sites, n_taxa), layer 0 is dosage (-1 = missing)
        tax_list (list): individual IDs aligned to genotype_dat axis 2
        ind_map (dict): individual metadata with 'population' and 'ploidy' keys
        site_df (pd.DataFrame): must contain 'chromosome', 'position', 'ref_allele', 'alt_allele'

    Returns:
        locus_labels (list[str]): locus names as 'chromosome_position'
        data_rows (list[list]): each row is [individual_id, pop_number, allele_code, ...]
    """
    n_sites = genotype_dat.shape[1]
    tax_index = {tax: idx for idx, tax in enumerate(tax_list)}

    # Build per-site allele codes
    ref_codes = np.full(n_sites, MISSING_CODE, dtype=np.int8)
    alt_codes = np.full(n_sites, MISSING_CODE, dtype=np.int8)
    for i in range(n_sites):
        ref = site_df.iloc[i]['ref_allele']
        alt = site_df.iloc[i]['alt_allele']
        if ref in NUCLEOTIDE_CODE:
            ref_codes[i] = NUCLEOTIDE_CODE[ref]
        if alt in NUCLEOTIDE_CODE:
            alt_codes[i] = NUCLEOTIDE_CODE[alt]

    # Locus labels
    locus_labels = [
        f"{site_df.iloc[i]['chromosome']}_{site_df.iloc[i]['position']}"
        for i in range(n_sites)
    ]

    # Population ordering: sorted alphabetically, 1-based numbering
    populations = assign_populations(ind_map)
    sorted_pops = sorted(populations.keys())
    pop_number = {pop: idx + 1 for idx, pop in enumerate(sorted_pops)}

    # Order individuals by sorted population, then by VCF order within population
    ordered_individuals = []
    for pop in sorted_pops:
        pop_members = populations[pop]
        for tax in tax_list:
            if tax in pop_members:
                ordered_individuals.append(tax)

    # Build data rows
    dosage_layer = genotype_dat[0]
    data_rows = []
    for individual in ordered_individuals:
        tidx = tax_index[individual]
        ploidy = int(ind_map[individual]['ploidy'])
        pop_num = pop_number[ind_map[individual]['population']]

        for k in range(ploidy):
            row = [individual, pop_num]
            for site in range(n_sites):
                dosage = int(dosage_layer[site, tidx])
                if dosage < 0:
                    row.append(MISSING_CODE)
                else:
                    row.append(int(alt_codes[site]) if k < dosage else int(ref_codes[site]))
            data_rows.append(row)

    return locus_labels, data_rows


def write_structure_file(locus_labels, data_rows, output_path):
    """
    Write a STRUCTURE-format file.

    Parameters:
        locus_labels (list[str]): locus names for the header row
        data_rows (list[list]): data rows from build_structure_matrix
        output_path (str): path to write the output file
    """
    with open(output_path, 'w') as fh:
        fh.write('\t\t' + '\t'.join(locus_labels) + '\n')
        for row in data_rows:
            fh.write('\t'.join(str(v) for v in row) + '\n')
