from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data
import numpy as np
# Reload and preprocess data
X_pca, y, selected_features, explained_variance = load_and_preprocess_data('data.csv')

# Split data
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


