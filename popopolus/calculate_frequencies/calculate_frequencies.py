import re
import numpy as np
import pandas as pd
import logging


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_sample_fields(format_string, sample_string):
    """Extract GT/AD/GQ fields from a sample column using FORMAT keys."""
    format_keys = format_string.split(':')
    sample_values = sample_string.split(':')
    field_map = {}
    for idx, key in enumerate(format_keys):
        if idx < len(sample_values):
            field_map[key] = sample_values[idx]

    gt_field = field_map.get('GT', sample_values[0] if len(sample_values) > 0 else '')
    ad_field = field_map.get('AD', sample_values[1] if len(sample_values) > 1 else '')
    gq_field = field_map.get('GQ', '')
    return gt_field, ad_field, gq_field, ('GQ' in field_map)


def get_ind_genotypes(n_sites, n_tax, ind_map, vcf_file, min_depth, min_count, min_qual, pass_flag, output_dir, return_site_data=False):
    '''
    Returns per-individual genotype dosage and quality layers across sites from a multisample VCF.

    Parameters:
        n_sites (int): the number of sites to process from the VCF
        n_tax (int): the number of individuals to process from the VCF
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier 
        vcf_file (string): a multisample vcf file uncompressed
        min_depth (int): the minimum depth of a site to be considered high-quality
        min_count (int): the minimum number of reads supporting the minor allele to be considered high-quality
        min_qual (int): the minimum phred-scaled genotype likelihood to be considered high-quality
        pass_flag (bool): Does the VCF have an PASS field or something else to condider like a "."
        output_dir (string): the directory where all results will be written

    Returns:
        tax_list (list): A list of individual labels
        genotype_dat: a numpy array with shape (4, n_sites, n_tax) where layers are genotype dosage,
            depth, genotype quality, and pass-filter indicator
        site_df (pd.DataFrame, optional): site-level coordinates and chromosome ids when return_site_data=True
    '''

    genotype_data = np.empty((n_sites, n_tax), dtype=np.int8)
    site_depth_data = np.empty((n_sites, n_tax), dtype=np.uint16)
    genotype_quality_data = np.empty((n_sites, n_tax), dtype=np.uint8)
    passing_filter_data = np.empty((n_sites, n_tax), dtype=np.bool_)
    chromosome_data = np.empty((n_sites,), dtype=np.uint16)
    site_position_data = np.empty((n_sites,), dtype=np.uint32)
    chromosome_label_data = np.empty((n_sites,), dtype=object)
    ref_allele_data = np.empty((n_sites,), dtype='U1')
    alt_allele_data = np.empty((n_sites,), dtype='U1')
    vcf_map = {}
    vcf_index = {}
    tax_list = []
    n_tax = 0
    n_sites = 0
    n_variants = {}
    skip_header = 1

    chromosome_map = {}
    chromosome_count = 0

    with open(vcf_file,'r') as fh:
        for line in fh:
            line = line.strip()
            if '#CHROM' in line:
                temp = line.split()
                #print(line)
                for i in range(9, len(temp)):
                    this_tax = ''
                    if '/' in temp[i]:
                        tax_path = temp[i].split('/')
                        this_tax = tax_path[-1]
                    else:
                        this_tax = temp[i]
                    #print(this_tax)
                    if this_tax in ind_map.keys():
                        tax_list.append(this_tax)
                        n_variants[this_tax] = 0
                        vcf_index[this_tax] = n_tax
                        n_tax = n_tax + 1
                        vcf_map[i] = this_tax
                skip_header = 0
            else:
                if skip_header == 0:
                    temp = line.split()
                    if (temp[6] == pass_flag):
                        format_string = temp[8]
                        for i in range(9, len(temp)): ####Continue fixing here
                            if i in vcf_map.keys():
                                genotypes = []
                                genotype_sum = -1
                                ref_counts = 0
                                alt_counts = 0
                                total_count = 0
                                genotype_quality = 0
                                indicator = 0
                                genotype_string = temp[i]
                                gt_field, ad_field, gq_field, has_gq = _extract_sample_fields(format_string, genotype_string)
                                if re.match(r'\d+,\d+', ad_field):
                                    allele_counts = ad_field.split(',')
                                    ref_counts = _safe_int(allele_counts[0], default=0)
                                    alt_counts = _safe_int(allele_counts[1], default=0)
                                    total_count = ref_counts + alt_counts
                                if re.match(r'\d+\S+\d+', gt_field):
                                        if '/' in gt_field:
                                            genotypes = gt_field.split('/')
                                        elif '|' in gt_field:
                                            genotypes = gt_field.split('|')
                                        genotype_sum = int(genotypes.count('1'))
                                if re.match(r'\d+', gq_field):
                                    genotype_quality = min(_safe_int(gq_field, default=0), np.iinfo(np.uint8).max)
                                qual_ok = (genotype_quality >= min_qual) if has_gq else True
                                if ((total_count >= min_depth) and (ref_counts >= 1) and (alt_counts >= min_count) and qual_ok):
                                    indicator = 1

                                genotype_data[n_sites, vcf_index[vcf_map[i]]] = genotype_sum
                                site_depth_data[n_sites, vcf_index[vcf_map[i]]] = total_count
                                genotype_quality_data[n_sites, vcf_index[vcf_map[i]]] = genotype_quality
                                passing_filter_data[n_sites, vcf_index[vcf_map[i]]] = indicator
                                n_variants[vcf_map[i]] = n_variants[vcf_map[i]] + 1
                        chromosome = temp[0]
                        position = temp[1]
                        ref_allele_data[n_sites] = temp[3]
                        alt_allele_data[n_sites] = temp[4].split(',')[0]
                        if chromosome not in chromosome_map.keys():
                            chromosome_map[chromosome] = chromosome_count
                            chromosome_count = chromosome_count + 1
                            chromosome_data[n_sites] = int(chromosome_map[chromosome])
                        elif chromosome in chromosome_map.keys():
                            chromosome_data[n_sites] = int(chromosome_map[chromosome])
                        chromosome_label_data[n_sites] = chromosome
                        site_position_data[n_sites] = int(position)
                        n_sites = n_sites + 1

    if (output_dir != 'dummy'):
        for i in range(0, len(tax_list)):
            output_file = f'{output_dir}/{tax_list[i]}.txt'
            outfile = open(output_file, 'w')
            outfile.write('genotype\tdepth\tgenotype_quality\tpass_filters\n')
            for j in range(0, n_variants[tax_list[i]]):
                outstring = f'{genotype_data[j,vcf_index[tax_list[i]]]}\t{site_depth_data[j,vcf_index[tax_list[i]]]}\t{genotype_quality_data[j,vcf_index[tax_list[i]]]}\t{passing_filter_data[j,vcf_index[tax_list[i]]]}\n'
                outfile.write(outstring)
            outfile.close()
    
    genotype_dat = np.array([
        genotype_data,
        site_depth_data,
        genotype_quality_data,
        passing_filter_data
    ])
    
    logging.info(f'Array shape: {genotype_dat.shape}')
    logging.info(f'Memory usage: {genotype_dat.nbytes / 1024 / 1024:.2f} MB')
    logging.info(f'Processed VCF of {n_sites} for {n_tax}\n')
    if return_site_data:
        site_df = pd.DataFrame(
            {
                'site_index': np.arange(0, n_sites, dtype=np.int32),
                'chromosome': chromosome_label_data[:n_sites],
                'chromosome_id': chromosome_data[:n_sites],
                'position': site_position_data[:n_sites],
                'ref_allele': ref_allele_data[:n_sites],
                'alt_allele': alt_allele_data[:n_sites],
            }
        )
        site_df['chromosome'] = site_df['chromosome'].astype('category')
        return(tax_list, genotype_dat, site_df)
    return(tax_list, genotype_dat)

def get_ind_ab(n_sites, n_tax, ind_map, vcf_file, min_depth, min_count, min_qual, pass_flag, output_dir, return_site_data=False):
    '''
    Returns per-individual allele-balance and quality layers across sites from a multisample VCF.

    Parameters:
        n_sites (int): the number of sites to process from the VCF
        n_tax (int): the number of individuals to process from the VCF
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier 
        vcf_file (string): a multisample vcf file uncompressed
        min_depth (int): the minimum depth of a site to be considered high-quality
        min_count (int): the minimum number of reads supporting the minor allele to be considered high-quality
        min_qual (int): the minimum phred-scaled genotype likelihood to be considered high-quality
        pass_flag (bool): Does the VCF have an PASS field or something else to condider like a "."
        output_dir (string): the directory where all results will be written

    Returns:
        tax_list (list): A list of individual labels
        ab_dat: a numpy array with shape (4, n_sites, n_tax) where layers are allele balance,
            depth, genotype quality, and pass-filter indicator
        site_df (pd.DataFrame, optional): site-level coordinates and chromosome ids when return_site_data=True
    '''
    
    # Goal - these all need to be typed as arrays to keep the memory from exploding
    # The individual files can be written out using pandas from array
    allele_balance_data = np.empty((n_sites, n_tax), dtype=np.float32)
    site_depth_data = np.empty((n_sites, n_tax), dtype=np.uint16)
    genotype_quality_data = np.empty((n_sites, n_tax), dtype=np.uint8)
    passing_filter_data = np.empty((n_sites, n_tax), dtype=np.bool_)
    chromosome_data = np.empty((n_sites,), dtype=np.uint16)
    site_position_data = np.empty((n_sites,), dtype=np.uint32)
    chromosome_label_data = np.empty((n_sites,), dtype=object)
    vcf_map = {}
    vcf_index = {}
    tax_list = []
    n_tax = 0
    n_sites = 0
    n_variants = {}
    skip_header = 1

    chromosome_map = {}
    chromosome_count = 0

    with open(vcf_file,'r') as fh:
        for line in fh:
            line = line.strip()
            if '#CHROM' in line:
                temp = line.split()
                #print(line)
                for i in range(9, len(temp)):
                    this_tax = ''
                    if '/' in temp[i]:
                        tax_path = temp[i].split('/')
                        this_tax = tax_path[-1]
                    else:
                        this_tax = temp[i]
                    #print(this_tax)
                    if this_tax in ind_map.keys():
                        tax_list.append(this_tax)
                        n_variants[this_tax] = 0
                        vcf_index[this_tax] = n_tax
                        n_tax = n_tax + 1
                        vcf_map[i] = this_tax
                skip_header = 0
            else:
                if skip_header == 0:
                    temp = line.split()
                    if (temp[6] == pass_flag):
                        format_string = temp[8]
                        for i in range(9, len(temp)): ####Continue fixing here
                            if i in vcf_map.keys():
                                ref_counts = 0
                                alt_counts = 0
                                total_count = 0
                                total_count = 0
                                allele_balance = 0
                                genotype_quality = 0
                                indicator = 0
                                genotype_string = temp[i]
                                _, ad_field, gq_field, has_gq = _extract_sample_fields(format_string, genotype_string)
                                if re.match(r'\d+,\d+', ad_field):
                                    allele_counts = ad_field.split(',')
                                    ref_counts = _safe_int(allele_counts[0], default=0)
                                    alt_counts = _safe_int(allele_counts[1], default=0)
                                    total_count = ref_counts + alt_counts
                                    if re.match(r'\d+', gq_field):
                                        genotype_quality = min(_safe_int(gq_field, default=0), np.iinfo(np.uint8).max)
                                    if (total_count > 0):
                                        allele_balance = alt_counts / total_count
                                    qual_ok = (genotype_quality >= min_qual) if has_gq else True
                                    if ((total_count >= min_depth) and (ref_counts >= 1) and (alt_counts >= min_count) and qual_ok):
                                        indicator = 1
                                else:
                                    print(f'WARNING: Incorrectly formatted VCF fields!\n--> {vcf_map[i]} at variant {n_variants[vcf_map[i]]}\n-->{temp[0]}: {temp[1]}\n')
                                
                                allele_balance_data[n_sites, vcf_index[vcf_map[i]]] = allele_balance
                                site_depth_data[n_sites, vcf_index[vcf_map[i]]] = total_count
                                genotype_quality_data[n_sites, vcf_index[vcf_map[i]]] = genotype_quality
                                passing_filter_data[n_sites, vcf_index[vcf_map[i]]] = indicator
                                n_variants[vcf_map[i]] = n_variants[vcf_map[i]] + 1
                        chromosome = temp[0]
                        position = temp[1]
                        if chromosome not in chromosome_map.keys():
                            chromosome_map[chromosome] = chromosome_count
                            chromosome_count = chromosome_count + 1
                            chromosome_data[n_sites] = int(chromosome_map[chromosome])
                        elif chromosome in chromosome_map.keys():
                            chromosome_data[n_sites] = int(chromosome_map[chromosome])
                        chromosome_label_data[n_sites] = chromosome
                        site_position_data[n_sites] = int(position)
                        n_sites = n_sites + 1

    if (output_dir != 'dummy'):
        for i in range(0, len(tax_list)):
            output_file = f'{output_dir}/{tax_list[i]}.txt'
            outfile = open(output_file, 'w')
            outfile.write('allele_balance\tdepth\tgenotype_quality\tpass_filters\n')
            for j in range(0, n_variants[tax_list[i]]):
                outstring = f'{allele_balance_data[j,vcf_index[tax_list[i]]]}\t{site_depth_data[j,vcf_index[tax_list[i]]]}\t{genotype_quality_data[j,vcf_index[tax_list[i]]]}\t{passing_filter_data[j,vcf_index[tax_list[i]]]}\n'
                outfile.write(outstring)
            outfile.close()
    
    ab_dat = np.array([
        allele_balance_data,
        site_depth_data,
        genotype_quality_data,
        passing_filter_data
    ])
    
    #ab_df = pd.DataFrame(allele_balance_data)
    #print(ab_df)
    logging.info(f'Array shape: {ab_dat.shape}')
    logging.info(f'Memory usage: {ab_dat.nbytes / 1024 / 1024:.2f} MB')
    logging.info(f'Processed VCF of {n_sites} for {n_tax}\n')
    if return_site_data:
        site_df = pd.DataFrame(
            {
                'site_index': np.arange(0, n_sites, dtype=np.int32),
                'chromosome': chromosome_label_data[:n_sites],
                'chromosome_id': chromosome_data[:n_sites],
                'position': site_position_data[:n_sites],
            }
        )
        site_df['chromosome'] = site_df['chromosome'].astype('category')
        return(tax_list, ab_dat, site_df)
    return(tax_list, ab_dat)


def get_pop_freqs(genotype_dat, tax_list, ind_map, site_df):
    '''
    Calculate population-level allele frequencies from individual genotype dosages.

    Parameters:
        genotype_dat (np.ndarray): shape (4, n_sites, n_taxa), layer 0 is dosage, layer 3 is pass-filter
        tax_list (list): individual IDs aligned to genotype_dat axis 2
        ind_map (dict): individual metadata with 'population' and 'ploidy' keys
        site_df (pd.DataFrame): site-level coordinates with 'chromosome' and 'position' columns

    Returns:
        freq_df (pd.DataFrame): wide matrix with populations as rows and locus labels as columns
    '''
    from popopolus.utils import assign_populations

    tax_index = {tax: idx for idx, tax in enumerate(tax_list)}
    populations = assign_populations(ind_map)
    sorted_pops = sorted(populations.keys())

    n_sites = genotype_dat.shape[1]
    dosage_layer = genotype_dat[0]
    pass_layer = genotype_dat[3]

    locus_labels = [
        f"{site_df.iloc[i]['chromosome']}_{site_df.iloc[i]['position']}"
        for i in range(n_sites)
    ]

    freq_data = {}
    for pop in sorted_pops:
        members = populations[pop]
        member_indices = np.array([tax_index[tax] for tax in members if tax in tax_index])
        ploidies = np.array([int(ind_map[tax]['ploidy']) for tax in members if tax in tax_index])

        pop_dosages = dosage_layer[:, member_indices].astype(np.float64)
        pop_passing = pass_layer[:, member_indices].astype(np.float64)

        # Only count individuals that pass filters at each site
        valid_dosages = np.where(pop_passing > 0, pop_dosages, 0.0)
        valid_ploidies = pop_passing * ploidies[np.newaxis, :]

        sum_dosages = valid_dosages.sum(axis=1)
        sum_ploidies = valid_ploidies.sum(axis=1)

        with np.errstate(invalid='ignore'):
            freqs = np.where(sum_ploidies > 0, sum_dosages / sum_ploidies, np.nan)
        freq_data[pop] = freqs

    freq_df = pd.DataFrame(freq_data, index=locus_labels).T
    freq_df.index.name = 'population'
    return freq_df
