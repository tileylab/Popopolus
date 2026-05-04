import numpy as np
import tempfile
from ppgtk.fit_mixtures.gmm import fit_gmm_to_ab
from ppgtk.fit_mixtures.lmm import fit_mixed_model_ab

def test_fit_gmm_to_ab():
    """
    Test that primary optimization function can return expected results from some simulated data
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ploidy_results = []
        np.random.seed(3232)
        ab_left = np.random.normal(0.25, 0.05, (500, 4))
        ab_middle = np.random.normal(0.5, 0.05, (1000, 4))
        ab_right = np.random.normal(0.75, 0.05, (500,4))
        allele_balance_array = np.concatenate([ab_left, ab_middle, ab_right], axis=0)
        #print(allele_balance_array)
        allele_mask_array = np.random.randint(0, 2, size=(2000, 4))
        ab_dat = np.array([allele_balance_array,allele_mask_array])
        #print(ab_dat)

        for i in range(len(ab_dat[0,0,:])):
            print(i)
            ind_dat = ab_dat[0,:,i]
            #print(ind_dat)
            ind_mask = (ab_dat[1,:,i] == 1)
            #print(ind_mask)
            ind_dat_filtered = ind_dat[ind_mask].reshape(-1, 1)
            best_n, predictions = fit_gmm_to_ab(ind_name = f'{i}', dat = ind_dat_filtered, ploidy = [2,3,4,5,6], model_constraints = 1, output_dir = temp_dir)
            ploidy_results.append(best_n)
        assert ploidy_results == [4,4,4,4]

#Place - holder function until working out some bugs
def test_fit_mixed_model_ab():
    """
    Test that linear mixed model can correctly detect random effects
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        lnLs = []
        pvalchecks = []
        np.random.seed(3232)
        ab_left = np.random.normal(0.25, 0.05, (500, 4))
        ab_middle = np.random.normal(0.5, 0.05, (1000, 4))
        ab_right = np.random.normal(0.75, 0.05, (500,4))
        allele_balance_array = np.concatenate([ab_left, ab_middle, ab_right], axis=0)
        allele_depth_array = np.random.randint(20, 100, size=(2000, 4))
        allele_mask_array = np.random.randint(0, 2, size=(2000, 4))
        ab_dat = np.array([allele_balance_array,allele_depth_array,allele_mask_array])
        for i in range(len(ab_dat[0,0,:])):
            ind_dat = ab_dat[0,:,i]
            ind_depth = ab_dat[1,:,i]
            ind_mask = (ab_dat[2,:,i] == 1)
            ind_dat_filtered = ind_dat[ind_mask]
            ind_depth_filtered = ind_depth[ind_mask]
            dat = ind_dat_filtered.reshape(-1,1)
            best_n, predictions = fit_gmm_to_ab(ind_name = f'{i}', dat = dat, ploidy = [2,3,4,5,6], model_constraints = 1, output_dir = temp_dir)
            lmm_result, rand_effects, fixed_effects, p_value = fit_mixed_model_ab(ind_name = f'{i}', allele_balance_data = ind_dat_filtered, site_depth_data = ind_depth_filtered, gmm_predictions = predictions, output_dir = temp_dir)
            lnLs.append(int(lmm_result.llf))
            if (p_value < 0.001):
                pvalchecks.append(1)
            else:
                pvalchecks.append(0)
        assert lnLs == [-3736,-3728,-3716,-3825]
        assert pvalchecks == [1,1,1,1]

