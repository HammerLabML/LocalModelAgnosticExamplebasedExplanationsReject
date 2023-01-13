from sklearn.preprocessing import StandardScaler


def scale_standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
