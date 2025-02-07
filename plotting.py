import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def plot_trace(chains, parameter_names=None, thin=10):
    num_params = chains[0].shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), squeeze=False)
    for i in range(num_params):
        ax = axes[i, 0]
        for chain_idx, chain in enumerate(chains):
            ax.plot(chain[::thin, i], label=f'Chain {chain_idx+1}', alpha=0.6)
        param_name = parameter_names[i] if parameter_names else f'PC {i}'
        ax.set_title(f'Trace Plot for {param_name}')
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_posterior_hist(samples, parameter_names=None):
    num_params = samples.shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), squeeze=False)
    for i in range(num_params):
        ax = axes[i, 0]
        ax.hist(samples[:, i], bins=30, density=True)
        param_name = parameter_names[i] if parameter_names else f'PC {i}'
        ax.set_title(f'Posterior Distribution for {param_name}')
    plt.tight_layout()
    plt.show()

def plot_autocorrelation(samples, parameter_names=None, max_lag=100, thin=10):
    num_params = samples.shape[1]
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 2 * num_params), squeeze=False)
    for i in range(num_params):
        ax = axes[i, 0]
        plot_acf(samples[::thin, i], lags=max_lag, ax=ax)
        param_name = parameter_names[i] if parameter_names else f'PC {i}'
        ax.set_title(f'Autocorrelation for {param_name}')
    plt.tight_layout()
    plt.show()
