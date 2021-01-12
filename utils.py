import os

#directory of folder where article files stored
def create_corpus(directory):
    corpus = []
    for filename in os.listdir(directory):
        try:
            article_content = open(os.path.join(directory, filename), 'r').read()

            corpus.append(article_content)
        except:
            pass
            #print("Exception in file:", filename)
    return corpus

import nltk
import heapq

def create_bagofwords(corpus):
    wordfreq = {}
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    most_freq = heapq.nlargest(1000, wordfreq, key=wordfreq.get)
    sentence_vectors = []
    for sentence in corpus:
        sentence_tokens = nltk.word_tokenize(sentence)
        if len(sentence_tokens) > 20:
            sent_vec = []
            for token in most_freq:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)

    sentence_vectors = np.asarray(sentence_vectors)
    return sentence_vectors

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

        # find sorted indeces of eig_vals array in descending order
        descending_indices = list(np.flip(np.argsort(eig_vals)))

        # find n best eigen values indices
        best_indices = descending_indices[:self.n]

        #extract corresponding vectors
        V = eig_vectors[:,best_indices]

        projection = np.dot(X, V)
        
        return projection.astype('float64')

def processed_data(data):

    # removing everything except alphabets`
    proc_data = data.str.replace("[^a-zA-Z#]", " ")

    # removing short words
    proc_data = proc_data.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    # make all text lowercase
    proc_data = proc_data.apply(lambda x: x.lower())
    
    stop_words = nltk.corpus.stopwords.words('english')

    # tokenization
    tokenized_doc = proc_data.apply(lambda x: x.split())

    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    # de-tokenization
    detokenized_doc = []
    for i in range(len(data)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    return detokenized_doc
