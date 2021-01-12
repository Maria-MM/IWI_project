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

my_pca = utils.MyPCA(10)
projected = my_pca.project_data(sentence_vectors)

pca = PCA(n_components=10)
projected2 = pca.fit_transform(sentence_vectors) 

print("projections using standard PCA implementation from sklearn\n", projected2[:3])

print("projections using own implementation\n", projected[:3])

