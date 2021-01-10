import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pca


#code for testing PCA on Another dataset
plt.style.use('ggplot')
# Load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Z-score the features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# The PCA model
standard_pca = PCA(n_components=2) # estimate only 2 PCs
X_new = standard_pca.fit_transform(X) # project the original data into the PCA space



mypca = pca.MyPCA(2)
X_new2 = mypca.project_data(X)

print("Is shape of both projections equal:",X_new.shape == X_new2.shape)
print(np.concatenate((X_new, X_new2), axis = 1)[:5])


fig, axes = plt.subplots(1,3)
axes[0].scatter(X[:,0], X[:,1], c=y)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
axes[2].scatter(X_new2[:,0], X_new2[:,1], c=y)
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].set_title('After my PCA')
plt.show()

