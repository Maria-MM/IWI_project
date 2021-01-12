import utils

#directory of folder where article files stored
directory = "../articles_processed"

corpus = utils.create_corpus(directory)

sentence_vectors = utils.create_bagofwords(corpus)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(sentence_vectors)
sentence_vectors = scaler.transform(sentence_vectors)

my_pca = utils.MyPCA(2)
projected = my_pca.project_data(sentence_vectors)

pca = PCA(n_components=2)
projected2 = pca.fit_transform(sentence_vectors) 

print("projections using standard PCA implementation from sklearn\n", projected2[:3])

print("projections using own implementation\n", projected[:3])

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2)
axes[0].scatter(projected[:,0], projected[:,1])
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('Our PCA')
axes[1].scatter(projected2[:,0], projected2[:,1])
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Standard PCA')
plt.show()
