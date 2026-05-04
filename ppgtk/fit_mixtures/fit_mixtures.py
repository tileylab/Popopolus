import numpy as np
import pandas as pd
import logging
from ppgtk.fit_mixtures.gmm import fit_gmm_to_ab
from ppgtk.fit_mixtures.lmm import fit_mixed_model_ab

####
# Main ppgtk function
# Consider moving out to other submodule
####
def est_ploidy(tax_list, ab_dat, method, ploidy_levels, minimum_sites, model_constraints, output_dir):
    """
    Estimate ploidy from allele balance data using the specified method.
    
    Parameters:
        tax_list (list): A list of individual names corresponding to the individual order of ab_dat
        ab_dat (np.array): Allele balance data returned from get_ind_freqs.
        method (str): Method for estimating ploidy ('gmm' or 'other').
        ploidy_levels (str): The ploidies to test passed as a comma-separated list.
        minimum_sites (int): The minimum number of sites to be considered for analysis
        model_constraints (int): The parameters to contrain where 0 is none, 1 is means, and 2 is means and weights
        output_dir (str): The output directory where all results will be directed
    
    Returns:
        ploidy_df: DataFrame containing estimated ploidy for each individual.
    """
    if method == 'gmm':
        output_file = f'{output_dir}/ploidy.txt'
        outfile = open(output_file, 'w')
        ploidy_dict = {}
        ploidy_level_list = ploidy_levels.split(',')
        ploidy = [int(p) for p in ploidy_level_list]
        logging.info(f'Testing for ploidy with the following values:\n{ploidy}\n')
        for i in range(len(ab_dat[0,0,:])):
            ind_name = tax_list[i]
            ind_dat = ab_dat[0,:,i]
            ind_depth = ab_dat[1,:,i]
            ind_mask = ab_dat[3,:,i] == 1
            ind_dat_filtered = ind_dat[ind_mask]
            ind_depth_filtered = ind_depth[ind_mask]
            ind_dat_buffer = (ind_dat_filtered > 0.05) & (ind_dat_filtered < 0.95)
            ind_dat_filtered_truncated = ind_dat_filtered[ind_dat_buffer]
            ind_depth_filtered_truncated = ind_depth_filtered[ind_dat_buffer]
            dat = ind_dat_filtered_truncated
            if len(ind_dat_filtered_truncated.shape) == 1:
                dat = ind_dat_filtered_truncated.reshape(-1, 1)
            # Fit GMM to allele balance data
            #print(f'min sites: {minimum_sites}\n')
            #print(f'{len(dat[:])}\t{dat.shape}\n')
            n_sites = len(dat[:])
            if n_sites >= minimum_sites:
                logging.info(f"Individual {ind_name}: {n_sites} sites")
                best_n, predictions = fit_gmm_to_ab(ind_name, dat, ploidy, model_constraints, output_dir)
                #print(predictions)
                #print(type(ind_dat_filtered_truncated))
                #print(type(ind_dat_filtered_truncated))
                #print(type(predictions))

                if best_n > 2:
                    lmm_result, rand_effects, fixed_effects, p_value = fit_mixed_model_ab(ind_name, ind_dat_filtered_truncated, ind_depth_filtered_truncated, predictions, output_dir)
                    print(lmm_result.summary())
                    print(rand_effects)
                    print(fixed_effects)
                    print(f'p-value versus diploid assumption: {p_value}')
                    #print(f'{best_n}')
                    outfile.write(f'{ind_name}\t{best_n}\t{p_value}\n')
                else:
                    print('diploid detected - skipping lmm')
                    outfile.write(f'{ind_name}\t{best_n}\tNA\n')
                ploidy_dict[ind_name] = best_n
            else:
                logging.warning(f'Individual {ind_name}: Sample skipped due to low site count passing filters.\n')
                ploidy_dict[ind_name] = None
        outfile.close()
        ploidy_df = pd.DataFrame.from_dict(ploidy_dict, orient = 'index')
        ploidy_df.reset_index(inplace=True)
        ploidy_df.columns = ['Individual','Ploidy']
        return(ploidy_df)
    else:
        logging.error('Terminated due to unavailable estimation method!\n')
        raise ValueError("Unsupported method. Use 'gmm'.")