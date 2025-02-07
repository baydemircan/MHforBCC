import numpy as np

def metropolis_hastings(log_posterior, initial_beta, iterations, X, y, initial_step_size=0.1, target_acceptance=0.234, return_acceptance=False):
    beta = initial_beta.copy()
    samples = []
    accepted = 0
    step_size = initial_step_size
    
    for i in range(iterations):
        # Propose a new beta
        beta_proposal = beta + np.random.normal(0, step_size, size=beta.shape)
        
        # Compute acceptance probability
        log_posterior_current = log_posterior(beta, X, y)
        log_posterior_proposal = log_posterior(beta_proposal, X, y)
        acceptance_prob = min(1, np.exp(log_posterior_proposal - log_posterior_current))
        
        # Accept or reject
        if np.random.rand() < acceptance_prob:
            beta = beta_proposal
            accepted += 1
        
        samples.append(beta.copy())
        
        # Adapt step size
        if i % 100 == 0 and i > 0:
            acceptance_rate = accepted / (i + 1)
            if acceptance_rate > target_acceptance:
                step_size *= 1.1
            else:
                step_size /= 1.1
    
    acceptance_rate = accepted / iterations
    print(f"Final step size: {step_size:.4f}")
    print(f"Acceptance Rate: {acceptance_rate:.2f}")
    
    if return_acceptance:
        return np.array(samples), acceptance_rate
    else:
        return np.array(samples)