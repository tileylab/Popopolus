import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import logging

def plot_gmm_fit_sklearn(data, gmm, output_dir, plot_name="result.fit.png", n_points=1000, title="GMM Fit to Data"):
    """
    Plot histogram of observed data with fitted GMM components from sklearn.
    
    Parameters:
        data (np.array): Original data used to fit the GMM
        gmm (GaussianMixtureModel): Fitted GMM model wrapper class
        n_points (int): Number of points for plotting the GMM curves
        title (str): Plot title
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of observed data
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Observed Data')
    
    # Generate points for plotting the GMM components
    x = np.linspace(min(data), max(data), n_points).reshape(-1, 1)
    
    # Plot individual components
    for i in range(gmm.n_components):
        # Get parameters for this component
        mu = gmm.means_[i][0]
        sigma = np.sqrt(gmm.covariances_[i][0][0])
        weight = gmm.weights_[i]
        
        # Calculate component distribution
        component = weight * norm.pdf(x, mu, sigma)
        plt.plot(x, component, '--', label=f'Component {i+1}')
    
    # Plot total mixture
    log_prob = gmm.score_samples(x)
    total = np.exp(log_prob)
    plt.plot(x, total, 'r-', label='Total Mixture', linewidth=2)
    
    plt.xlabel('Allele Balance')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/{plot_name}.png', dpi=300)
    plt.close('all')


def plot_lmm_fit(ind_name, alt_count_data, ref_count_data, site_class, lmm_result, output_dir):
    """
    Create scatter plot of allele balance vs depth, colored by GMM component,
    with fitted LMM regression lines and confidence intervals.
    
    Parameters:
        ind_name: Name of individual for plot title
        alt_count_data: Array of counts for alternate allele
        ref_count_data: Array of counts for reference allele
        site_class: Array of GMM component assignments
        lmm_result: Fitted statsmodels MixedLM object
        output_dir: Directory to save plot
    """
    print(f'Creating lmm figure for {ind_name} in {output_dir}')
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Alt Counts': alt_count_data,
        'Ref Counts': ref_count_data,
        'Component': [f'Component {x}' for x in site_class]})
    print(plot_df.head())

    # Get unique components in sorted order
    unique_components = sorted(plot_df['Component'].unique())
    n_colors = len(unique_components)

    # Create color palette with explicit ordering
    sns_colors = sns.color_palette('tab10', n_colors)
    component_colors = dict(zip(unique_components, sns_colors))

    # Create base scatter plot
    sns.scatterplot(data=plot_df, 
                    x='Alt Counts', 
                    y='Ref Counts',
                    hue='Component',
                    hue_order=unique_components,
                    palette=component_colors,
                    alpha=0.5)
    
    # Generate prediction lines
    ref_count_range = np.linspace(ref_count_data.min(), ref_count_data.max(), 100)
    
    # Get fixed effect coefficients
    intercept = lmm_result.fe_params['Intercept']
    ref_count_effect = lmm_result.fe_params['ref_counts']
    
    # Calculate confidence intervals
    conf_int = lmm_result.conf_int()
    lower_intercept, upper_intercept = conf_int.loc['Intercept']
    lower_ref_count, upper_ref_count = conf_int.loc['ref_counts']
    
    # Plot overall fixed effect line
    plt.plot(ref_count_range, 
            intercept + ref_count_effect * ref_count_range,
            'k-', label='Population Average', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(ref_count_range,
                    lower_intercept + lower_ref_count * ref_count_range,
                    upper_intercept + upper_ref_count * ref_count_range,
                    color='gray', alpha=0.2, label='95% CI')
    
    # Plot group-specific prediction lines
    #print(lmm_result.random_effects)
    if len(lmm_result.random_effects) > 1:
        # Color map is now created up above to ensure correct ordering of components by seaborn
        #n_colors = len(lmm_result.random_effects)
        #color_map = dict(zip(
        #    lmm_result.random_effects.keys(),
        #    sns_colors
        #))
        
        # Plot each group's prediction line
        for group, effect in lmm_result.random_effects.items():
            # Get random effect for this group
            #print(f'Successfull accessed random intercept for group {group}')
            #print(group)
            #print(effect)
            group_effect = effect.iloc[0]  # First value is the random intercept
                
            # Calculate group-specific prediction
            group_prediction = (intercept - group_effect) + ref_count_effect * ref_count_range
            #print(f'Group prediction is {group_prediction}')
            # Plot with color from map
            component_key = f'Component {group}'
            plt.plot(ref_count_range, group_prediction,
                    '--', color=component_colors[component_key], alpha=0.8,
                    label=f'{component_key} Prediction',
                    linewidth=1.5)
    
    # Adjust legend to show both points and lines
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.xlabel('Reference Allele Count')
    plt.ylabel('Alternate Allele Count')
    plt.title(f'Ref Vs. Alt Counts - {ind_name}')
    plt.legend(handles, labels, 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              title='Components')
    
    output_file = f'{output_dir}/{ind_name}.lmm.png'
    plt.savefig(output_file, 
                dpi=300, 
                bbox_inches='tight',  # This ensures the legend is included
                pad_inches=0.5)      # Add padding around the plot
    plt.close('all')
    logging.info("Scatterplot of ref versus alt counts saved successfully")