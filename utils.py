import os
import nltk
import heapq
import numpy as np

def create_name_to_article_dict(directory):
    name_to_article = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            try:
                article_content = open(os.path.join(directory, filename), 'r').read()
                name = filename.split('.')[0]
                name_to_article[int(name)] = article_content
            except:
                pass
                #print("Exception in file:", filename)
    return name_to_article


def create_corpus(dictionary, min_article_word_count = 20):
    corpus = []
    n = max(dictionary)
    deleted_ids = []

    for i in range(n):
        if i in dictionary:
            if len(nltk.word_tokenize(dictionary[i])) > min_article_word_count:
                corpus.append(dictionary[i])
            else:
                deleted_ids.append(i)

    return corpus, deleted_ids


def create_name_to_category_id_dict(list_filename, dict_filename):
    name_to_cat_id = {}
    category_list = open(list_filename, 'r').readlines()
    category_set = set([item.split()[0] for item in category_list])

    lines = open(dict_filename, 'r').readlines()
    n = len(lines)
    for i in range(n):
        items = lines[i].split()
        for item in items:
            if item in category_set:
                name_to_cat_id[i] = item
                break
    return name_to_cat_id


def create_article_cat_id_array(name_to_article, name_to_cat_id, deleted_ids):
    n = max(name_to_article)
    deleted_set = set(deleted_ids)
    cat_id_array = [int(name_to_cat_id[i]) for i in range(n) if (i in name_to_article and i not in deleted_set)]
    return cat_id_array


def simplify_id_array(arr):
    initial_to_simpler = {}
    current = 0
    new_arr = []
    for item in arr:
        if item in initial_to_simpler:
            new_arr.append(initial_to_simpler[item])
        else:
            initial_to_simpler[item] = current
            new_arr.append(current)
            current += 1
    return new_arr


def create_id_to_category_dict(filename):
    id_to_category = {}
    lines = open(filename, 'r').readlines()
    for line in lines:
        split = line.split()
        id_to_category[int(split[1])] = split[0]
    return id_to_category


def create_bagofwords(corpus, features_count = 1000):
    wordfreq = {}
    for article in corpus:
        tokens = nltk.word_tokenize(article)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    most_freq = heapq.nlargest(features_count, wordfreq, key=wordfreq.get)
    article_vectors = []

    for article in corpus:
        article_tokens = nltk.word_tokenize(article)
        art_vec = []
        for token in most_freq:
            if token in article_tokens:
                art_vec.append(1)
            else:
                art_vec.append(0)
        article_vectors.append(art_vec)

    article_vectors = np.asarray(article_vectors)

    return article_vectors


def apply_filter(array, filter_arr):
    filter_set = set(filter_arr)
    n = len(array)
    return [array[i] for i in range(n) if i not in filter_set]


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
