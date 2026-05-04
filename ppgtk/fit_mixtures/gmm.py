import numpy as np
import sys
from ppgtk.fit_mixtures.plot_mixtures import plot_gmm_fit_sklearn
from sklearn.mixture import GaussianMixture
from .gmm_fixed_means import GaussianMixtureFixedMeans
from .gmm_fixed_means_fixed_weights import GaussianMixtureFixedMeansFixedWeights

class GaussianMixtureModel:
    def __init__(self, n_components=1, covariance_type='full'):
        
        self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.score(X)

class GaussianMixtureModelFixedMeans:
    def __init__(self, n_components=1, covariance_type='full', means_init = np.array([0.5]).reshape(-1,1)):
        
        self.model = GaussianMixtureFixedMeans(n_components=n_components, covariance_type=covariance_type, means_init = means_init)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.score(X)
    
class GaussianMixtureModelFixedMeansFixedWeights:
    def __init__(self, n_components=1, covariance_type='full', means_init = np.array([0.5]).reshape(-1,1), weights_init = np.array([1.0]).reshape(-1,1)):
        
        self.model = GaussianMixtureFixedMeansFixedWeights(n_components=n_components, covariance_type=covariance_type, means_init = means_init, weights_init = weights_init)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.score(X)
    
def get_fixed_params(n_components):
    means = None
    if (n_components > 6):
        sys.ext('ERROR: Ploidy greater than 6 is not implemented or recommended! Stopping.')
    elif(n_components < 1):
        sys.exit('ERROR: The maximum ploidy must be a positive integer! Stopping.')
    else:
        if n_components == 1:
            means = np.array([0.5]).reshape(-1,1)
            weights = np.array([1.0])
        if n_components == 2:
            means = np.array([1/3,2/3]).reshape(-1,1)
            weights = np.array([0.5,0.5])
        if n_components == 3:
            means = np.array([0.25,0.5,0.75]).reshape(-1,1)
            weights = np.array([0.25,0.5,0.25])
        if n_components == 4:
            means = np.array([0.2,0.4,0.6,0.8]).reshape(-1,1)
            weights = np.array([0.25,0.25,0.25,0.25])
        if n_components == 5:
            means = np.array([1/6,2/6,0.5,4/6,5/6]).reshape(-1,1)
            weights = np.array([1/6,1/6,2/6,1/6,1/6])
    return(means, weights)

def fit_gmm_to_ab(ind_name, dat, ploidy, model_constraints, output_dir):
    """
    Fit Gaussian Mixture Model (GMM) to allele balance data.
    
    Parameters:
        ind_name (string): The name of the individual
        dat (np.array): Allele balance data.
        n_components (int): Number of components in the GMM.
    """
    # Fit GMM to allele balance data
    best_n = 1
    best_bic = np.inf
    best_gmm = None
    output_file = f'{output_dir}/{ind_name}.fit.txt'
    outfile = open(output_file, 'w')
    for i in range(0, len(ploidy)):
        n_components = ploidy[i] - 1
        gmm = None
        means, weights = get_fixed_params(n_components)
        if model_constraints == 0:
            gmm = GaussianMixture(n_components = n_components)
        if model_constraints == 1:
            gmm = GaussianMixtureFixedMeans(n_components = n_components, means_init = means)
        if model_constraints == 2:
            gmm = GaussianMixtureFixedMeansFixedWeights(n_components = n_components, means_init = means, weights_init = weights)
        gmm.fit(dat)
        score = gmm.score(dat)
        bic = gmm.bic(dat)
        outfile.write(f'Model for ploidy = {ploidy[i]}\n')
        outfile.write("Fitted GMM parameters:\n")
        outfile.write(f'Means:\n {gmm.means_}\n')
        outfile.write(f'Covariances:\n {gmm.covariances_}\n')
        outfile.write(f'Weights:\n {gmm.weights_}\n')
        # Print the best likelihood
        outfile.write(f'Best likelihood: {score}\n')
        outfile.write(f'BIC: {bic}\n')
        outfile.write('\n')
        # only consider a 3.2 point difference via Kass and Raftery 1995
        if bic < (best_bic - 3.2):
            best_bic = bic
            best_n = ploidy[i]
            best_gmm = gmm
        #We can return the categories for each point based on posterior probabilities too
        #Will be used in downstream linear models
        #Create permutation test to check if model is actually a good fit
        predictions = best_gmm.predict(dat)
        #print(predictions)
    outfile.close()
    plot_gmm_fit_sklearn(dat, best_gmm, output_dir, plot_name=f'{ind_name}.fit', title=f'GMM Fit to Allele Balance Data ({ind_name})')

    return(best_n, predictions)