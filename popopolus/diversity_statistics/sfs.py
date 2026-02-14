import re
import numpy as np
import pandas as pd
import logging

def initialize_sfs(populations):
    '''
    Initializes a site frequency spectrum (SFS) dictionary for each population.

    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    '''
    n_populations = len(populations.keys())
    # Hardcoding all individuals as diploid for now
    ploidy_factor = 2
    max_sample_size = 0
    for pop in populations.keys():
        n_haplotypes = 0
        for ind in populations[pop]:
            n_haplotypes = n_haplotypes + ploidy_factor
            # In the future, we will access the ploidy by individual
        if n_haplotypes > max_sample_size:
            max_sample_size = n_haplotypes
    sfs = np.zeros((n_populations, max_sample_size + 1), dtype=np.uint32)
    return(sfs)

def occupy_sfs(populations, tax_list, genotype_dat, sfs):
    '''
    Calculate the site frequency spectrum (SFS) for each population.

    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
        tax_list (list): A list of individual names corresponding to the individual order of genotype_dat
        genotype_dat (np.ndarray): a 2D numpy array of genotype data with shape (n_sites, n_individuals)
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    '''
    # Only concern ourselves with the global calulcation for now. Will add per-interval later.
    for pop_index, pop in enumerate(populations.keys()):
        inds_in_pop = populations[pop]
        ind_indices = [tax_list.index(ind) for ind in inds_in_pop if ind in tax_list]
        logging.info(f'Calculating SFS for population: {pop} with {len(ind_indices)} individuals\n')
        pop_genotype_dat = genotype_dat[0, :, ind_indices]
        logging.info(f'pop_genotype_dat shape: {pop_genotype_dat.shape}')
        for site_index in range(len(pop_genotype_dat[0,:])):
            genotypes = pop_genotype_dat[: , site_index]
            # Quick fix for missing data but address weighting by variable sample size later
            # The imputation step should be done in the get_ind_freqs function but for now we will just set missing data to 0 (homozygous reference) and not worry about the bias this may introduce
            genotypes[genotypes == -1] = 0
            # Count number of derived alleles (assuming 0, 1, 2 coding)
            n_derived = np.sum(genotypes)
            #print(n_derived)
            sfs[pop_index, n_derived] += 1
    logging.info(f'sfs:\n {sfs}')
    return(sfs)