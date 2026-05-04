import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, List, Dict
import logging

# These functions are not in use. Just a poor attempt to lift over the old R code to avoid scikit learn and allow more flexibility in model specification
# Leaving to return to in the future, but optimization is not correct
def fit_mixture_model(x: np.ndarray, n_components: int, model_type: str = 'normal') -> Dict:
    """
    Fit a mixture model to data
    
    Parameters:
        x: Input data
        n_components: Number of mixture components
        model_type: Type of mixture model ('normal', 'gamma', or 'lognormal')
        
    Returns:
        Dictionary containing fitted parameters and log likelihood
    """
    model_classes = {
        'normal': NormalMixture,
        'gamma': GammaMixture,
        'lognormal': LognormalMixture
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_classes[model_type](n_components)
    result = model.fit(x)
    
    return result


class MixtureModel:
    """Base class for mixture models with different component distributions"""
    
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.weights = None
        self.params = None
        
    def _component_pdf(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Calculate PDF for each component. To be implemented by subclasses."""
        raise NotImplementedError
        
    def _init_params(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize model parameters using data quantiles"""
        weights = np.ones(self.n_components) / self.n_components
        # Sort data and split into n_components parts
        x_sorted = np.sort(x)
        splits = np.array_split(x_sorted, self.n_components)
        means = [np.mean(split) for split in splits]
        stds = [np.std(split) for split in splits]
        params = np.concatenate([means, stds])
        return weights, params
    
    def _negative_log_likelihood(self, params: np.ndarray, x: np.ndarray) -> float:
        """Calculate negative log likelihood for optimization"""
        n_samples = len(x)
        reshaped_params = params.reshape(2, self.n_components)
        component_probs = self._component_pdf(x[:, np.newaxis], reshaped_params)
        mixture_probs = np.sum(self.weights[:, np.newaxis] * component_probs, axis=0)
        return -np.sum(np.log(mixture_probs + 1e-10))
    
    def fit(self, x: np.ndarray, n_init: int = 10) -> Dict:
        """Fit mixture model using multiple random initializations"""
        best_nll = np.inf
        best_result = None
        
        for _ in range(n_init):
            # Initialize parameters
            weights, init_params = self._init_params(x)
            self.weights = weights
            
            # Optimize
            result = minimize(
                self._negative_log_likelihood,
                init_params,
                args=(x,),
                method='Nelder-Mead'
            )
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
                self.params = result.x
        
        return {
            'weights': self.weights,
            'params': self.params.reshape(2, self.n_components),
            'log_likelihood': -best_nll
        }

class NormalMixture(MixtureModel):
    """Mixture of Normal distributions"""
    
    def _component_pdf(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        means = params[0]
        stds = np.abs(params[1])  # Ensure positive standard deviations
        return stats.norm.pdf(x, means, stds)

class GammaMixture(MixtureModel):
    """Mixture of Gamma distributions"""
    
    def _component_pdf(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        shapes = np.abs(params[0])  # Ensure positive shape parameters
        scales = np.abs(params[1])  # Ensure positive scale parameters
        return stats.gamma.pdf(x, shapes, scale=scales)

class LognormalMixture(MixtureModel):
    """Mixture of Lognormal distributions"""
    
    def _component_pdf(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        locs = params[0]
        scales = np.abs(params[1])  # Ensure positive scale parameters
        return stats.lognorm.pdf(x, scales, loc=0, scale=np.exp(locs))