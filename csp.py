import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh


    
    # def _compute_covariance_matrices(self, X, y):
    #     _, n_channels, _ = X.shape

    #     covs = []
    #     for cur_class in self._classes:
    #         """Concatenate epochs before computing the covariance."""
    #         x_class = X[y==cur_class]
    #         x_class = np.transpose(x_class, [1, 0, 2])
    #         x_class = x_class.reshape(n_channels, -1)
    #         cov_mat = np.cov(x_class)
    #         covs.append(cov_mat)
        
    #     return np.stack(covs)

# class CustomCSP(BaseEstimator, TransformerMixin):
#     def __init__(self, n_components=None):
#         self.n_components = n_components

#     def fit(self, X, y):
#         covs = np.array([np.cov(epoch, rowvar = False) for epoch in X])

#         class_0 = np.mean(covs[y == 0], axis=0)
#         class_1 = np.mean(covs[y == 1], axis=0)
#         W = class_0 - class_1

#         eigvals, eigvecs = eigh(W)

#         sorted_indices = np.argsort(eigvals)[::-1]
#         eigvals = eigvals[sorted_indices]
#         eigvecs = eigvecs[:, sorted_indices]

#         if self.n_components is not None:
#             eigvecs = eigvecs[:, :self.n_components]

#         self.filter_components_ = eigvecs
#         return self

#     def transform(self, X):
#         # Projection des données sur les composantes spatiales CSP
#         return np.array([np.dot(epoch, self.filter_components_ ) for epoch in X])

#     def fit_transform(self, X, y, **fit_params): 
#         return super().fit_transform(X, y=y, **fit_params)
