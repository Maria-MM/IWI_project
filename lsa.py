#import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
pd.set_option("display.max_colwidth", 200)
import utils


class MyLSA:

    def do_topic_modelling(self, corpus, topics_count = 5, top_words_count = 5):

        data = pd.DataFrame({'documents':corpus})
        data['documents'] = utils.processed_data(data['documents'])

        #create tf-idf matrix document-term
        vectorizer = TfidfVectorizer(stop_words='english', 
        max_features= 1000, # keep top 1000 terms 
        max_df = 0.5, 
        smooth_idf=True)

        X = vectorizer.fit_transform(data['documents'])

        print("matrix document-term shape", X.shape) 

        #do SVD decomposition
        svd_model = TruncatedSVD(n_components=topics_count, algorithm='randomized', n_iter=100, random_state=122)
        svd_model.fit(X)

        print(len(svd_model.components_))

        terms = vectorizer.get_feature_names()

        #create topic to term dict
        topic_to_terms = {}
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:top_words_count]
            topic_to_terms["Topic"+str(i)] = sorted_terms

        #create document to topics weights matrix
        document_to_topics = svd_model.fit_transform(X)

        return topic_to_terms, document_to_topics