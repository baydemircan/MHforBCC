import numpy as np
from data_preprocessing import load_and_preprocess_data
from model import logistic_regression_likelihood, prior
from metropolis_hastings import metropolis_hastings
from plotting import plot_trace, plot_posterior_hist, plot_autocorrelation
from diagnostics import gelman_rubin_statistic
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.utils import shuffle
def log_posterior(beta, X, y):
    return logistic_regression_likelihood(beta, X, y) + prior(beta)

def effective_sample_size(chain):
    n = len(chain)
    if n <= 1:
        return 1
    acf = np.correlate(chain - np.mean(chain), chain - np.mean(chain), mode='full')[n-1:]
    acf /= acf[0]
    
    neg_loc = np.where(acf < 0)[0]
    if len(neg_loc) > 0:
        cut_off = neg_loc[0]
    else:
        cut_off = len(acf)
    
    sum_rho = 1 + 2 * np.sum(acf[1:cut_off])
    
    return n / sum_rho

def main():
    # Load and preprocess data
    X_pca, y, selected_features, explained_variance = load_and_preprocess_data('data.csv')
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Add intercept term to training data
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    # Permutation Test: Shuffle labels and retrain
    y_permuted = shuffle(y, random_state=42)  # Shuffle target labels
    X_train_perm, X_test_perm, y_train_perm, y_test_perm = train_test_split(X_pca, y_permuted, test_size=0.2, random_state=42)

    # Train logistic regression on shuffled labels
    lr_perm = LogisticRegression(random_state=42)
    lr_perm.fit(X_train_perm, y_train_perm)
    y_pred_permuted = lr_perm.predict_proba(X_test_perm)[:, 1]

    # Compute AUC for permuted labels
    fpr_perm, tpr_perm, _ = roc_curve(y_test_perm, y_pred_permuted)
    auc_permuted = auc(fpr_perm, tpr_perm)
    print(f"Permutation Test AUC: {auc_permuted:.4f}")

    # Train a baseline dummy classifier
    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_clf.fit(X_train[:, 1:], y_train)  # Exclude intercept
    y_dummy_pred = dummy_clf.predict_proba(X_test[:, 1:])[:, 1]

    # Compute AUC for dummy classifier
    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, y_dummy_pred)
    dummy_auc = auc(dummy_fpr, dummy_tpr)
    print(f"Dummy Classifier AUC: {dummy_auc:.4f}")
    
    # Number of parameters
    num_params = X_train.shape[1]
    
    # MCMC settings
    num_chains = 4
    iterations = 20000
    burn_in = int(iterations / 2)
    
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Running fold {fold+1}/5")
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train.to_numpy()[train_index], y_train.to_numpy()[val_index]

        # Ensure the validation set has both classes
        unique_labels = np.unique(y_fold_val)
        if len(unique_labels) < 2:
            print(f"Warning: Validation fold {fold+1} has only one class ({unique_labels})")
            continue

        # Run multiple chains
        chains = []
        acceptance_rates = []
        for chain_idx in range(num_chains):
            print(f"Running chain {chain_idx+1}/{num_chains}")
            np.random.seed(chain_idx)  # For reproducibility
            initial_beta = np.random.normal(0, 1, size=num_params)
            
            # Run Metropolis-Hastings sampling
            samples, acceptance_rate = metropolis_hastings(
                log_posterior, initial_beta, iterations, X_fold_train, y_fold_train, 
                return_acceptance=True)
            
            acceptance_rates.append(acceptance_rate)
            
            # Discard burn-in samples
            samples_burned = samples[burn_in:]
            
            chains.append(samples_burned)
        
        # Compute Gelman-Rubin statistic
        gelman_rubin_values = gelman_rubin_statistic(chains)
        
        # Parameter names
        feature_names = ['Intercept'] + [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        print("\nGelman-Rubin Statistics:")
        for name, gr in zip(feature_names, gelman_rubin_values):
            print(f"{name}: {gr:.4f}")
        
        # Compute validation predictions
        beta_means = np.mean(np.vstack(chains), axis=0)
        z_val = X_fold_val @ beta_means
        p_val = 1 / (1 + np.exp(-z_val))

        # Ensure predictions are valid before computing AUC
        if len(p_val) > 0:
            fpr, tpr, _ = roc_curve(y_fold_val, p_val)
            fold_auc = auc(fpr, tpr)
            cv_aucs.append(fold_auc)
        else:
            print(f"Warning: No valid predictions for AUC calculation in fold {fold+1}")
    
    # Print cross-validation results
    if len(cv_aucs) > 0:
        print(f"\nCross-validation AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    else:
        print("\nCross-validation AUC could not be computed (no valid folds)")

    # Run a simple logistic regression for comparison
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train[:, 1:], y_train)  # Exclude intercept for sklearn
    y_pred_proba = lr.predict_proba(X_test[:, 1:])[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_proba)
    lr_auc = auc(lr_fpr, lr_tpr)
    print(f"\nSimple Logistic Regression Test Set AUC: {lr_auc:.4f}")

    # Plot posterior and MCMC diagnostics
    plot_trace(chains, parameter_names=feature_names, thin=10)
    plot_posterior_hist(np.vstack(chains), parameter_names=feature_names)
    plot_autocorrelation(np.vstack(chains), parameter_names=feature_names, thin=10)

if __name__ == "__main__":
    main()
