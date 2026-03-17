import re
import numpy as np
import pandas as pd
import logging

def initialize_unfolded_sfs(populations, ind_map):
    '''
    Initializes a site frequency spectrum (SFS) dictionary for each population. The SFS is unfolded or derived, which assumes 0 represents the ancestral state. The length of the 1-d unfolded SFS is equal to the number of chromosomes in the sample + 1 (to account for the 0 bin).
    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier as well as their ploidy
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    '''
    n_populations = len(populations.keys())
    max_sample_size = 0
    for pop in populations.keys():
        n_haplotypes = 0
        for ind in populations[pop]:
            ploidy = ind_map[ind]['ploidy']
            n_haplotypes = n_haplotypes + ploidy
        if n_haplotypes > max_sample_size:
            max_sample_size = n_haplotypes
    sfs = np.zeros((n_populations, max_sample_size + 1), dtype=np.uint32)
    return(sfs)

def initialize_folded_sfs(populations, ind_map):
    '''
    Initializes a site frequency spectrum (SFS) dictionary for each population. The SFS is folded, which assumes 0 represents the allele at highest frequency in the population. The length of the 1-d folded SFS is equal to the number of chromosomes in the sample // 2 + 1 (to account for the 0 bin).
    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier as well as their ploidy
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    '''
    n_populations = len(populations.keys())
    max_sample_size = 0
    for pop in populations.keys():
        n_haplotypes = 0
        for ind in populations[pop]:
            ploidy = ind_map[ind]['ploidy']
            n_haplotypes = n_haplotypes + ploidy
        if (n_haplotypes//2) > max_sample_size:
            max_sample_size = n_haplotypes//2
    sfs = np.zeros((n_populations, max_sample_size + 1), dtype=np.uint32)
    return(sfs)


def occupy_unfolded_sfs(populations, tax_list, genotype_dat, sfs):
    '''
    Calculate the site frequency spectrum (SFS) for each population.

    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
        tax_list (list): A list of individual names corresponding to the individual order of genotype_dat
        genotype_dat (np.ndarray): a 3D numpy array with shape (n_layers, n_sites, n_individuals);
            layer 0 stores genotype dosage values used to construct the SFS
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size + 1)
    '''
    # Only concern ourselves with the global calulcation for now. Will add per-interval later.
    for pop_index, pop in enumerate(populations.keys()):
        inds_in_pop = populations[pop]
        ind_indices = [tax_list.index(ind) for ind in inds_in_pop if ind in tax_list]
        logging.info(f'Calculating SFS for population: {pop} with {len(ind_indices)} individuals\n')
        pop_genotype_dat = genotype_dat[0][:, ind_indices]
        logging.info(f'pop_genotype_dat shape: {pop_genotype_dat.shape}')
        for site_index in range(pop_genotype_dat.shape[0]):
            genotypes = pop_genotype_dat[site_index, :].copy()
            # Quick fix for missing data but address weighting by variable sample size later
            # The imputation step should be done in the get_ind_freqs function but for now we will just set missing data to 0 (homozygous reference) and not worry about the bias this may introduce
            genotypes[genotypes == -1] = 0
            # Count number of derived alleles (assuming 0, 1, 2 coding)
            n_derived = np.sum(genotypes)
            #print(n_derived)
            sfs[pop_index, n_derived] += 1
    logging.info(f'sfs:\n {sfs}')
    return(sfs)

def occupy_folded_sfs(populations, tax_list, genotype_dat, sfs):
    '''
    Calculate the folded site frequency spectrum (SFS) for each population. The folded SFS assumes that the major allele is the ancestral state and the minor allele is the derived state. This is accomplished by summing the counts at each site and taking the minimum of the count and the total number of chromosomes minus the count rather than modifying the genotype matrix itself.

    Parameters:
        populations (dict): a dict of unique populations as keys and individuals in those populations as a list as values
        tax_list (list): A list of individual names corresponding to the individual order of genotype_dat
        genotype_dat (np.ndarray): a 3D numpy array with shape (n_layers, n_sites, n_individuals);
            layer 0 stores genotype dosage values used to construct the SFS
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size//2 + 1)
    Returns:
        sfs (ndarray): a numpy array of 1-d site frequency spectra for each population with shape (n_populations, max_sample_size//2 + 1)
    '''
    # Only concern ourselves with the global calulcation for now. Will add per-interval later.
    for pop_index, pop in enumerate(populations.keys()):
        inds_in_pop = populations[pop]
        ind_indices = [tax_list.index(ind) for ind in inds_in_pop if ind in tax_list]
        logging.info(f'Calculating SFS for population: {pop} with {len(ind_indices)} individuals\n')
        pop_genotype_dat = genotype_dat[0][:, ind_indices]
        logging.info(f'pop_genotype_dat shape: {pop_genotype_dat.shape}')
        for site_index in range(pop_genotype_dat.shape[0]):
            genotypes = pop_genotype_dat[site_index, :].copy()
            # Quick fix for missing data but address weighting by variable sample size later
            # The imputation step should be done in the get_ind_freqs function but for now we will just set missing data to 0 (homozygous reference) and not worry about the bias this may introduce
            genotypes[genotypes == -1] = 0
            # Count number of derived alleles (assuming 0, 1, 2 coding)
            n_derived = np.sum(genotypes)
            # Current genotype encoding is diploid dosage (0, 1, 2), so each sample contributes two chromosomes.
            n_chromosomes = len(genotypes) * 2
            n_folded = min(n_derived, n_chromosomes - n_derived)
            #print(n_derived)
            sfs[pop_index, n_folded] += 1
    logging.info(f'sfs:\n {sfs}')
    return(sfs)