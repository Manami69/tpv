import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh


class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        X_1 = X[y == 0]
        X_2 = X[y == 1]
   
        C_1 = self._get_2D_cov(X_1)
        C_2 = self._get_2D_cov(X_2)

        eigvals, eigvecs = eigh(C_1, C_1 + C_2)

        sorted_indices = np.argsort(np.abs(eigvals - 0.5))[::-1]
        eigvecs = eigvecs[:, sorted_indices]

        if self.n_components is not None:
            eigvecs = eigvecs[:, :self.n_components]
        self.filter_components_ = eigvecs

    def transform(self, X):
        """Compute CSP filter on X Datas and returns the power of
            CSP features averaged over time and shape"""
        X = np.array([np.dot(self.filter_components_.T, epoch) for epoch in X])
        X = (X**2).mean(axis=2)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _get_2D_cov(self, X_class):
        n_channels = X_class.shape[1]
        X_class = np.transpose(X_class, [1, 0, 2])
        X_class = X_class.reshape(n_channels, -1)
        return np.cov(X_class)
