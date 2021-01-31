import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import lsa

base_directory = "./dump_out/"
directory = base_directory + "temp/temp"
topic_dict_path = base_directory + "temp-po_kompresji-categories-simple-20120104"
topic_list_path = base_directory + "temp-cats_links-simple-20120104"
category_to_id_path = base_directory + "temp-po_kompresji-cats_dict-simple-20120104"

if __name__ == '__main__':

    name_to_article = utils.create_name_to_article_dict(directory)
    corpus, deleted_ids = utils.create_corpus(name_to_article)
    category_dict = utils.create_name_to_category_id_dict(topic_list_path, topic_dict_path)
    cat_id_array = utils.create_article_cat_id_array(name_to_article, category_dict, deleted_ids)
    simple_id_array = utils.simplify_id_array(cat_id_array)
    category_dict = utils.create_id_to_category_dict(category_to_id_path)

    print("documents count", len(corpus))

    # do LSA
    my_lsa = lsa.MyLSA()
    topic_to_terms, document_to_topics = my_lsa.do_topic_modelling(corpus, topics_count = 5)

    print("\nTopics:")
    for k,v in topic_to_terms.items():
        print(k)
        for item in v:
            print(item[0], ":", item[1])
        print("\n")

    topic_strings = ["'Topic %i'" % i for i in range(len(topic_to_terms))]
    document_descriptions = []
    count = 0

    for document in document_to_topics:

        weight_topic = zip(document, topic_strings)
        document_descriptions.append("document %i: " % count + "".join("%.2f * %s  " % (item[0], item[1]) for item in weight_topic))
        count += 1

    print(document_descriptions[:5])

    category_to_best_match_count = {}
    category_to_count = {}
    count = 0
    #evaluate the results, compare to real category
    for document in document_to_topics:
        #best_match_topic_id = document.index(max(document))
        best_match_topic_id = np.argmax(document, axis=0)
        cat_name = category_dict[cat_id_array[count]]

        if cat_name not in category_to_best_match_count:
            category_to_best_match_count[cat_name] = {}

        if cat_name not in category_to_count:
            category_to_count[cat_name] = 1
        else:
            category_to_count[cat_name] += 1

        if best_match_topic_id in category_to_best_match_count[cat_name]:
            category_to_best_match_count[cat_name][best_match_topic_id] += 1
        else:
            category_to_best_match_count[cat_name][best_match_topic_id] = 1

        count += 1

    print("\nResult validation:")

    for k, v in category_to_best_match_count.items():
        print("Category", k, "documents best matches (maximal topic weights) are:")
        doc_count = category_to_count[k]
        for k1,v1 in v.items():
            print("Topic", k1, "in", v1, "cases out of", doc_count, ", percentage:", v1/doc_count*100);