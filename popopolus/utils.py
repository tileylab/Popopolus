import pandas as pd
import os
import logging
import sys

def check_dir(my_dir):
    if os.path.exists(my_dir):
        print(f'Existing output directory found. Writing output to: {my_dir}\n')
    else:
        os.makedirs(my_dir)
        print(f'Output files will be written to: {my_dir}\n')

def decompress_vcf(vcf_file):
    print(f'{vcf_file}\n')

def map_individuals(sample_sheet):
    '''
    Returns a dict mapping individual ids in the vcf to populations. Confirms individuals are present in VCF and warns if missing.

    Parameters:
        sample_sheet (string): a sample sheet mapping individual ids to individual names and population names
        vcf (string): a multisample vcf file uncompressed

    Returns:
        ind_map (dict): a dictionary where keys are ids in the vcf and values are preferred names in downstream output
    '''
    df = pd.read_csv(sample_sheet,header=0)
    logging.info('')
    logging.info(df.head())
    ind_map = df.set_index('individual').to_dict('index')
    #print(ind_map)
    return(ind_map)

def assign_populations(ind_map):
    '''
    Returns a list of unique populations from the ind_map.

    Parameters:
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier
    Returns:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
    '''
    populations = {}
    for ind in ind_map.keys():
        pop = ind_map[ind]['population']
        if pop not in populations.keys():
            populations[pop] = [ind]
        else:
            populations[pop].append(ind)
    return (populations)

def get_vcf_dimensions(vcf_file, pate_flag, ind_map):
    """
    Get the number of individuals and number of sites from the vcf in order to get dimensions for allocation of numpy arrays.
    Perform an additional check to ensure that all individuals specified in the ind_map are present in the vcf.
    """
    n_tax = 0
    n_sites = 0
    skip_header = 1
    tax_list = []
    with open(vcf_file,'r') as fh:
        for line in fh:
            line = line.strip()
            if '#CHROM' in line:
                temp = line.split()
                #print(line)
                for i in range(9, len(temp)):
                    this_tax = ''
                    if '/' in temp[i]:
                        if pate_flag == False:
                            tax_path = temp[i].split('/')
                            this_tax = tax_path[-1]
                        elif pate_flag == True:
                            tax_path = temp[i].split('/')
                            this_tax = tax_path[-2]
                    else:
                        this_tax = temp[i]
                    #print(this_tax)
                    tax_list.append(this_tax)
                    if this_tax in ind_map.keys():
                        n_tax = n_tax + 1
                skip_header = 0
            else:
                if skip_header == 0:
                    temp = line.split()
                    if (temp[6] == 'PASS' or ((pate_flag == True) and temp[6] == '.')):
                        n_sites = n_sites + 1
    if n_tax < len(ind_map.keys()):
        logging.warning('Not all indivuals in mapping file are present in VCF. Checking for mismatches...')
        for i in ind_map.keys():
            if i not in tax_list:
                logging.warning(f'{i} found in mapping file but not VCF!')
        logging.error('Stopping to make corrections to mapping file!')
        sys.exit()
    logging.info(f'Found {n_sites} sites and {n_tax} individuals')
    return n_sites,n_tax
