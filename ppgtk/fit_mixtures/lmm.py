import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from scipy.stats import chi2
from ppgtk.fit_mixtures.plot_mixtures import plot_lmm_fit

####
# Fit a linear mixed model to ab as a function of depth to assess model fit
####
def fit_mixed_model_ab(ind_name: str,
                       allele_balance_data: np.ndarray, 
                      site_depth_data: np.ndarray, 
                      gmm_predictions: np.ndarray,
                      output_dir: str):
    """
    Fit separate linear mixed models for each individual's allele balance data
    using depth as fixed effect and GMM component assignments as random effects.
    
    Parameters:
        allele_balance_data: np.array of allele balance values per individual
        site_depth_data: np.array of read depths per individual
        gmm_predictions: np.array of GMM component assignments per individual
    
    Returns:
        Tuple of (model_result, random_effects, fixed_effects, p_value)
    """
    # Create DataFrame that includes the 
    alt_count_data = np.array(allele_balance_data * site_depth_data)
    #print(alt_count_data)
    ref_count_data = np.array(site_depth_data - alt_count_data)
    #print(ref_count_data)
    df = pd.DataFrame({
        'allele_balance': allele_balance_data[:],
        'depth': site_depth_data[:],
        'alt_counts': alt_count_data[:],
        'ref_counts': ref_count_data[:],
        'site_class': pd.Categorical(gmm_predictions[:])
    })
        
    # Remove any rows with NaN values
    df = df.dropna()
                
    # Fit model for this individual
    # Random effect is now just component-based
    model = smf.mixedlm(
        "alt_counts ~ ref_counts",
        data=df,
        groups="site_class",
        re_formula = "~1"
    )
        
    result = model.fit()
    random_effects = result.random_effects
    fixed_effects = result.fe_params
    #print(type(result))
    plot_lmm_fit(ind_name, alt_count_data, ref_count_data, gmm_predictions, result, output_dir)

    # Test if the group effect is significantly better than no group effect
    # This is to beat down fitting of noise
    null_model = smf.ols(
        "alt_counts ~ ref_counts",
        data=df
    )
    null_result = null_model.fit()
    
    print(f"Full Model Log-Likelihood: {result.llf}")
    print(f"Restricted Model Log-Likelihood: {null_result.llf}")
    
    lrt_statistic = -2 * (null_result.llf - result.llf)

    # Calculate degrees of freedom (difference in number of parameters)
    # This requires knowing how many parameters differ between the two models
    degrees_of_freedom = result.df_modelwc - null_result.df_model

    # Calculate p-value
    p_value = chi2.sf(lrt_statistic, degrees_of_freedom)

    print(f"\nLR Statistic: {lrt_statistic}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {degrees_of_freedom}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("\nReject the null hypothesis. The full model is a significantly better fit.")
    else:
        print("\nFail to reject the null hypothesis. The restricted model is sufficient.")

    
    return(result, random_effects, fixed_effects, p_value)

