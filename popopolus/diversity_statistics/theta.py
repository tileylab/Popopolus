import numpy as np
import pandas as pd
import logging
from popopolus.utils import assign_populations
from popopolus.math import combination
from popopolus.utils import count_chromosomes
from popopolus.diversity_statistics.sfs import initialize_unfolded_sfs
from popopolus.diversity_statistics.sfs import occupy_unfolded_sfs
from popopolus.diversity_statistics.sfs import initialize_folded_sfs
from popopolus.diversity_statistics.sfs import occupy_folded_sfs

def estimate_thetas(genotype_dat, tax_list, ind_map, intervals, folded, output_dir):
    '''
    Returns a pandas DataFrame object of theta across each interval by population defined in the ind_map.

    Parameters:
        genotype_data (np.ndarray): a 3D numpy array with shape (n_layers, n_sites, n_individuals)
            where layer 0 stores genotype dosage values used for SFS/theta calculations
        tax_list (list): A list of individual names corresponding to the individual order of genotype_dat
        ind_map (dict): a dictionary mapping individuals in the VCF to a population or other identifier 
        intervals (list): a list of tuples defining the start and end positions of intervals to calculate theta over
        folded (bool): whether to calculate folded or unfolded site frequency spectrum
        output_dir (string): the directory where all results will be written

    Returns:
        theta_df: a pandas DataFrame of theta values across each interval by population
    '''
    logging.info(f'Estimating Watterson Theta:\n')
    populations = assign_populations(ind_map)
    if folded:
        sfs = initialize_folded_sfs(populations, ind_map)
        sfs = occupy_folded_sfs(populations, tax_list, genotype_dat, sfs)
    else:
        sfs = initialize_unfolded_sfs(populations, ind_map)
        sfs = occupy_unfolded_sfs(populations, tax_list, genotype_dat, sfs)

    # Watterson's theta is calculated as S / a1 where S is the number of segregating sites and
    # a1 is the sum of 1/i for i from 1 to n-1 where n is the number of sampled chromosomes.
    # Ignoring variable ploidy among individuals for now
    # Ignoring intervals for now
    theta_results = []
    for pop_index, pop in enumerate(populations.keys()):
        n_individuals = len(populations[pop])
        #print(ind_map)
        #print(populations[pop])
        ind_list = populations[pop]
        # n_chromosomes is the true sampled chromosome count (n in theta formulas).
        n_chromosomes = count_chromosomes(ind_list, ind_map)
        #n_chromosomes = len(derived_sfs[pop_index, :]) # Number of sfs bins should be correct regardless of ploidy. This would be equal to the number of individuals * 2 + 1 for a diploid population
        #print(f'Diversity Estimates for {pop}: {n_individuals} individuals and {n_chromosomes - 1} chromosomes\n')
        if folded:
            # Folded SFS contains bins 0..floor(n/2). Bin 0 is monomorphic-major.
            max_folded_bin = n_chromosomes // 2
            segregating_bins = sfs[pop_index, 1:max_folded_bin + 1]
            allele_counts = np.arange(1, max_folded_bin + 1)
        else:
            # Unfolded SFS contains bins 0..n. Bins 1..n-1 are segregating.
            segregating_bins = sfs[pop_index, 1:n_chromosomes]
            allele_counts = np.arange(1, n_chromosomes)

        S = np.sum(segregating_bins) # Number of segregating sites is the sum of bins 1..n-1 (or folded equivalent)
        n_sites = np.sum(sfs[pop_index, :]) # Total number of sites is the sum across all bins
        
        # Normalization constants
        a1 = np.sum(1 / np.arange(1, n_chromosomes)) # a1 is used to normalize wattersons theta. The other terms are used in the tajima D normalization constant.
        a2 = np.sum(1 / np.arange(1, n_chromosomes)**2)
        b1 = (n_chromosomes + 1) / (3 * (n_chromosomes - 1))
        b2 = 2 * (n_chromosomes**2 + n_chromosomes + 3)/(9 * n_chromosomes * (n_chromosomes - 1))

        c1 = b1 - 1/a1
        c2 = b2 - (n_chromosomes + 2)/(a1 * n_chromosomes) + a2 / a1**2

        e1 = c1 / a1
        e2 = c2 / (a1**2 + a2)
        Vd = e1 * S + e2 * S * (S - 1)
        #----------------#
        theta_pi = 1/(combination(n_chromosomes, 2)) * np.sum(allele_counts * (n_chromosomes - allele_counts) * segregating_bins)
        theta_pi_persite = theta_pi / n_sites
        theta_wattersons = S / a1
        theta_wattersons_persite = theta_wattersons / n_sites
        tajima_D = (theta_pi - theta_wattersons) / np.sqrt(Vd)
        theta_results.append({'population': pop, 'n_individuals': n_individuals, 'n_chromosomes': n_chromosomes, 'theta_wattersons': theta_wattersons_persite, 'theta_pi': theta_pi_persite, 'tajima_D': tajima_D})
    theta_df = pd.DataFrame(theta_results)
    theta_df.to_csv(f'{output_dir}/theta.csv', index=False)
    return(theta_df)