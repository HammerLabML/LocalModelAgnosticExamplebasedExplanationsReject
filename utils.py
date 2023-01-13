import os
import random
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from data_preparation import scale_standardize_data

reject_thresholds = [0.001, 0.005, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 7, 10, 100, 1000]

knn_parameters = {'n_neighbors': [3, 5, 7, 10, 15]}
random_forest_parameters = {
    'n_estimators': [20, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'random_state': [444]
}

n_folds = 5
non_zero_threshold = 1e-5
non_zero_threshold_sparsity = 1e-5

def load_data(data_desc, data_folder="data/", scaling=True):
    if data_desc == "breastcancer":
        X, y = load_breast_cancer(return_X_y=True)
        if scaling is True:
            X = scale_standardize_data(X)
        return X, y
    elif data_desc == "wine":
        X, y = load_wine(return_X_y=True)
        if scaling is True:
            X = scale_standardize_data(X)
        return X, y
    elif data_desc == "flip":
        flip_data = np.load(os.path.join(data_folder, f"flip.npz"))
        if scaling is False:
            flip_data = np.load(os.path.join(data_folder, f"flip_notscaled.npz"))
        return flip_data['X'], flip_data['y']
    elif data_desc == "t21":
        t21_data = np.load(os.path.join(data_folder, f"t21.npz"))
        if scaling is False:
            t21_data = np.load(os.path.join(data_folder, f"t21_notscaled.npz"))
        return t21_data['X'], t21_data['y']
    else:
        raise ValueError(f"Unkown data set {data_desc}")



def evaluate_sparsity(xcf, x_orig):
    return evaluate_sparsity_ex(xcf - x_orig)

def evaluate_sparsity_ex(x):
    return np.sum(np.abs(x[i]) > non_zero_threshold_sparsity for i in range(x.shape[0])) / len(x)    # Count non-zero features (smaller values are better!)


def select_random_feature_subset(n_features, size=0.3):
    n_subset_size = int(n_features * size)

    return random.sample(range(n_features), n_subset_size)


def apply_perturbation(X, features_idx, noise_size=1.):
    scale = noise_size  # Scale/amount/variance of noise
    X[:, features_idx] += np.random.normal(scale=scale, size=(X.shape[0], len(features_idx)))
    return X


def evaluate_perturbed_features_recovery(xcf, x_orig, perturbed_features_idx):
    return evaluate_perturbed_features_recovery_ex(np.abs(xcf - x_orig), perturbed_features_idx)

def evaluate_perturbed_features_recovery_ex(x, perturbed_features_idx):
    indices = np.argwhere(x > non_zero_threshold)
    e = 1e-7

    # Compute confusion matrix
    tp = np.sum([idx in perturbed_features_idx for idx in indices]) / len(indices)
    fp = np.sum([idx not in perturbed_features_idx for idx in indices]) / len(indices)
    tn = np.sum([idx not in perturbed_features_idx for idx in filter(lambda i: i not in indices, range(x.shape[0]))]) / (len(list(filter(lambda i: i not in indices, range(x.shape[0])))) + e)
    fn = np.sum([idx in perturbed_features_idx for idx in filter(lambda i: i not in indices, range(x.shape[0]))]) / (len(list(filter(lambda i: i not in indices, range(x.shape[0])))) + e)

    if len(indices) != 0:
        return tp / (tp + fn + e)  # Compute recall
    else:
        return 0
