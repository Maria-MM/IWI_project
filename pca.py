import numpy as np
import warnings
warnings.filterwarnings("ignore")

class MyPCA:

    def __init__(self, n):

        self.n = n

    def project_data(self, X):

        # examples count
        m = X.shape[0]

        # find covariation matrix
        cov_matrix = 1 / m * np.dot(X.T, X)

        # find eigen values and vectors of cov_matrix
        eig_vals, eig_vectors = np.linalg.eig(cov_matrix)

        # find sorted indices of eig_vals array in descending order
        descending_indices = list(np.flip(np.argsort(eig_vals)))

        # find n best eigen values indices
        best_indices = descending_indices[:self.n]

        #extract corresponding vectors
        V = eig_vectors[:,best_indices]

        projection = np.dot(X, V)
        
        return projection.astype('float64')