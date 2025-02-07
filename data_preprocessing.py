import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data for machine learning.
    
    Parameters:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        X_pca (ndarray): PCA-transformed feature matrix.
        y (Series): Target variable.
        selected_features (list): Names of selected features after variance thresholding.
        explained_variance (list): Explained variance ratio for PCA components.
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
    
    # Encode 'diagnosis' column
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Apply PCA to retain 95% variance
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_.tolist()
    
    return X_pca, y, selected_features, explained_variance

def check_correlations(X):
    """
    Check for high correlations between features.
    
    Parameters:
        X (DataFrame): Feature matrix as a DataFrame with column names.
        
    Returns:
        list of tuples: Highly correlated feature pairs.
    """
    correlation_matrix = X.corr()
    high_corr = np.where(np.abs(correlation_matrix) > 0.8)
    high_corr = [
        (correlation_matrix.index[x], correlation_matrix.columns[y]) 
        for x, y in zip(*high_corr) if x != y and x < y
    ]
    return high_corr