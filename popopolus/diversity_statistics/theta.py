import re
import numpy as np
import pandas as pd
import logging
from popopolus.utils import assign_populations
from popopolus.diversity_statistics.sfs import initialize_sfs
from popopolus.diversity_statistics.sfs import occupy_sfs

def estimate_wattersons(genotype_dat, tax_list, ind_map, intervals, output_dir):
    '''
    Returns a pandas DataFrame object of theta across each interval by population defined in the ind_map.

    Parameters:
        genotype_data (np.ndarray): a 2D numpy array of genotype data with shape (n_sites, n_individuals)
        tax_list (list): A list of individual names corresponding to the individual order of genotype_dat
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier 
        intervals (list): a list of tuples defining the start and end positions of intervals to calculate theta over
        output_dir (string): the directory where all results will be written

    Returns:
        theta_df: a pandas DataFrame of theta values across each interval by population
    '''
    logging.info(f'Estimating Watterson Theta:\n')
    populations = assign_populations(ind_map)
    sfs = initialize_sfs(populations)
    derived_sfs = occupy_sfs(populations, tax_list, genotype_dat, sfs)

    # Watterson's theta is calculated as S / a1 where S is the number of segregating sites and a1 is the sum of 1/i for i from 1 to n-1 where n is the number of sites sampled.
    # Ignoring variable ploidy among individuals for now
    # Ignoring intervals for now
    theta_results = []
    for pop_index, pop in enumerate(populations.keys()):
        n_individuals = len(populations[pop])
        n_chromosomes = 2 * n_individuals
        S = np.sum(derived_sfs[pop_index, 1:n_chromosomes]) # Number of segregating sites is the sum of the SFS from 1 to n-1
        n_sites = np.sum(derived_sfs[pop_index, :]) # Total number of sites is the sum of the SFS from 0 to n
        a1 = np.sum(1 / np.arange(1, n_chromosomes))
        theta = S / a1
        theta_persite = theta / n_sites
        theta_results.append({'population': pop, 'theta': theta_persite})
    theta_df = pd.DataFrame(theta_results)
    theta_df.to_csv(f'{output_dir}/wattersons_theta.csv', index=False)
    return(theta_df)