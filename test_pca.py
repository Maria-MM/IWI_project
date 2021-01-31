from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import utils
import pca


base_directory = "./dump_out/"
directory = base_directory + "temp/temp"
topic_dict_path = base_directory + "temp-po_kompresji-categories-simple-20120104"
topic_list_path = base_directory + "temp-cats_links-simple-20120104"
category_to_id_path = base_directory + "temp-po_kompresji-cats_dict-simple-20120104"


if __name__ == '__main__':

    name_to_article = utils.create_name_to_article_dict(directory)
    corpus, deleted_ids = utils.create_corpus(name_to_article)

    name_to_cat_dict = utils.create_name_to_category_id_dict(topic_list_path, topic_dict_path)
    article_vectors= utils.create_bagofwords(corpus)
    cat_id_array = utils.create_article_cat_id_array(name_to_article, name_to_cat_dict, deleted_ids)
    simple_id_array = utils.simplify_id_array(cat_id_array)
    category_dict = utils.create_id_to_category_dict(category_to_id_path)

    # do feature scaling
    scaler = StandardScaler()
    scaler.fit(article_vectors)
    article_vectors = scaler.transform(article_vectors)

    my_pca = pca.MyPCA(2)
    projected = my_pca.project_data(article_vectors)

    standard_pca = PCA(n_components=2)
    projected2 = standard_pca.fit_transform(article_vectors) 

    print("projections using standard PCA implementation from sklearn\n", projected2[:3])

    print("projections using own implementation\n", projected[:3])

    #plot the results
    fig, axes = plt.subplots(1,2)
    group = np.array([category_dict[item] for item in cat_id_array])

    axes[0].scatter(projected2[:,0], projected2[:,1], c = simple_id_array)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Standard PCA')
    for g in np.unique(group):
        i = np.where(group == g)
        axes[0].scatter(projected2[i,0], projected2[i,1], label=g)
    axes[0].legend()

    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Our PCA')
    group = np.array([category_dict[item] for item in cat_id_array])
    for g in np.unique(group):
        i = np.where(group == g)
        axes[1].scatter(projected[i,0], projected[i,1], label=g)
    axes[1].legend()

    plt.show()
