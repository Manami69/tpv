import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        # Séparation des données en deux classes
        class_1 = X[y == 0]
        class_2 = X[y == 1]

        # Calcul des matrices de covariance pour chaque classe
        cov_1 = np.cov(class_1, rowvar=False)
        cov_2 = np.cov(class_2, rowvar=False)

        # Calcul de la matrice CSP
        eigvals, eigvecs = eigh(cov_1, cov_1 + cov_2)

        # Tri des vecteurs propres par ordre décroissant des valeurs propres
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Réduction dimensionnelle si spécifié
        if self.n_components is not None:
            eigvecs = eigvecs[:, :self.n_components]

        self.csp_components_ = eigvecs

    def transform(self, X):
        # Projection des données sur les composantes spatiales CSP
        return np.dot(X, self.csp_components_)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
