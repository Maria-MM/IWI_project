import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)

import utils

corpus = utils.create_corpus("../articles_processed")

print("documents count", len(corpus))

data = pd.DataFrame({'documents':corpus})

data['documents'] = utils.processed_data(data['documents'])

topics_count = 7

#create tf-idf matrix document-term
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(data['documents'])

print("matrix document-term shape", X.shape) 

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=topics_count, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

print(len(svd_model.components_))

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:5]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t)


document_topic = svd_model.fit_transform(X)

topic_strings = ["'Topic %i'" % i for i in range(topics_count)]
document_descriptions = []
count = 0
for document in document_topic:
    weight_topic = zip(document, topic_strings)
    document_descriptions.append("document %i: " % count + "".join("%f.2 * %s  " % (item[0], item[1]) for item in weight_topic))
    count += 1

print(document_descriptions[:5])

'''
my_pca = utils.MyPCA(2)
projected = my_pca.project_data(document_topic)

plt.scatter(projected[:, 0], projected[:, 1])
plt.show()
'''


