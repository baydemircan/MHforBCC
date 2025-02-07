
import numpy as np

def gelman_rubin_statistic(chains):
    """
    Computes the Gelman-Rubin statistic for MCMC chains.
    
    Parameters:
    chains : list of numpy arrays
        Each element in the list is a numpy array of shape (num_samples, num_params)
        representing the samples from one chain.
        
    Returns:
    R_hat : numpy array
        Gelman-Rubin statistics for each parameter.
    """
    M = len(chains)  
    N = chains[0].shape[0]  
    
    
    chain_means = np.array([np.mean(chain, axis=0) for chain in chains])
    
    
    overall_mean = np.mean(chain_means, axis=0)
    
    
    B = N / (M - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)  
    
    
    W = np.mean([np.var(chain, axis=0, ddof=1) for chain in chains], axis=0) 
    
    
    var_theta = (N - 1) / N * W + B / N
    
    
    R_hat = np.sqrt(var_theta / W)
    
    return R_hat