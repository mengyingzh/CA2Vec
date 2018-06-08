import numpy as np
import logging
import collections
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

logger = logging.getLogger('analyze_patient_icd.py')


def read_patient_icd_label(filename):
    with open(filename, "rb") as fname7:
        obj1 = np.load(fname7)
        visit_info = obj1.tolist()  # separator '0', label for one visit
        patient_icd_label = list()
        patient_icd_label.append(visit_info[0])
        for icd_index in range(1, len(visit_info)):
            if visit_info[icd_index] == 0:
                    patient_icd_label.append(visit_info[icd_index-1])
    return patient_icd_label


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")

    patient_icd_filename = ".\data\d2diag.np"
    patient_icd_label_list = read_patient_icd_label(patient_icd_filename)
    # logger.info(patient_icd_label_list)
    count = []
    n_class = 1000
    count.extend(collections.Counter(patient_icd_label_list).most_common(n_class))
    # logger.info(count)
    diagnose_set = set()
    for i in count:
        diagnose_set.add(i[0])

    # path_patient_vector = r"D:\Pycharmworkspace\word2vec\data\sae_data\sae_vector_100_patient.pickle"
    path_patient_vector = r"D:\Pycharmworkspace\word2vec\data\glove_data\glove_vector_100_patient.pickle"

    with open(path_patient_vector, 'rb') as pkl_file:
        patient_vector_list = pickle.load(pkl_file, encoding='bytes')
    logger.info("vector len %s" % len(patient_vector_list))

    patient_test = []
    patient_test_label = []

    for index, p_ele in enumerate(patient_vector_list[:n_class*4]):
        if diagnose_set.__contains__(patient_icd_label_list[index]):
            patient_test.append(p_ele)
            patient_test_label.append(patient_icd_label_list[index])

    logger.info("vector len %s" % len(patient_test))

    # evaluate embedding cluster result
    true_label = patient_test_label

    kmeans_model = KMeans(init='k-means++', n_clusters=n_class, n_init=n_class)
    kmeans_model.fit(patient_test)
    cluster_labels = kmeans_model.labels_
    logger.info("NMI: %s" % normalized_mutual_info_score(true_label, cluster_labels))
