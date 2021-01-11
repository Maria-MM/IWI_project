import nltk
import heapq
import numpy as np
import random
import string

import bs4 as bs
import urllib.request
import re


'''
raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:
    article_text += para.text

corpus = nltk.sent_tokenize(article_text)

for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])

#print(len(corpus))
#print(corpus[0])
'''
import os
corpus = []

#directory of folder where article files stored
directory = "../articles_processed"
for filename in os.listdir(directory):
    try:
        article_content = open(os.path.join(directory, filename), 'r').read()
        corpus.append(article_content)
    except:
        print("Exception in file:", filename)

print(len(corpus))
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

most_freq = heapq.nlargest(2000, wordfreq, key=wordfreq.get)
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

import pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(sentence_vectors)
sentence_vectors = scaler.transform(sentence_vectors)

my_pca = pca.MyPCA(10)
projected = my_pca.project_data(sentence_vectors)

pca = PCA(n_components=10) # estimate only 2 PCs
projected2 = pca.fit_transform(sentence_vectors) 

print("projections using standard PCA implementation from sklearn\n", projected2[:3])

print("projections using own implementation\n", projected[:3])
