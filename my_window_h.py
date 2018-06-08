"""Revised basic word2vec example for clinical activity vector representation learning."""

# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xlrd
import collections
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from collections import defaultdict
from collections import Counter
from sklearn.manifold import TSNE
import os
import logging
logger = logging.getLogger("my_skip_gram_basic")

plt.rcParams['font.sans-serif'] = ['SimHei']  # display Chinese Character


def read_data(filename):
    """Extract the first file as a list of words."""
    item_sub = []
    item_dosage_sub = []
    item_number_per_visit = []

    patient_num = 0
    count_total_visit = 0

    with open(filename, "r") as f:
        line = f.readline().strip('\n')
        while line:
            if not line.startswith("-1"):
                count_total_visit += 1
                ele = line.split(',')
                temp_list = [_.split(':')[0] for _ in ele]
                random.shuffle(temp_list)
                # logger.info(temp_list)
                item_number_per_visit.append(len(temp_list))
                for _ in temp_list:
                    item_sub.append(_)
            else:
                patient_num += 1
            line = f.readline().strip('\n')
    return item_sub, item_dosage_sub, patient_num, item_number_per_visit


def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]  # map of words(strings_id) to count of occurrences
    count.extend(collections.Counter(words).most_common(4216))  # never let out 1 word
    # logger.info("line 81 %s" % count)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)  # word, index: map of words(strings_id) to their codes(integers_id)
    # logger.info("line 84 %s" % len(dictionary))

    # data - list of codes (integers from 0 to vocabulary_size-1).
    #   This is the original text but words are replaced by their codes
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # index to word

    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size):
    global data_index
    global visit_index
    # assert batch_size % num_skips == 0
    # assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    visit_item_num_nearest = window_size_list[visit_index]
    visit_index += 1
    skip_window = max(1, visit_item_num_nearest//2)
    while batch_size % skip_window != 0:
        skip_window -= 1
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    num_skips = max(2, skip_window)  # number_skips: How many times to reuse an input to generate a label.
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0
        visit_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span
    temp = window_size_list[visit_index]
    # logger.info(buffer)
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]  # exclude center word index
        words_to_use = random.sample(context_words, num_skips)  # maximum number of words to use is num_skips
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            # buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
            temp -= 1
            if temp <= 0:
                visit_item_num_nearest += 1
                temp = window_size_list[visit_index]

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    visit_index = visit_index % len(window_size_list)
    # logger.info("batch %s" % batch)
    # logger.info("batch size %s" % len(batch))
    # logger.info("labels %s" % labels)
    # logger.info("labels size %s" % len(labels))
    return batch, labels


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")
    # start pregram timing

    # Step 0: Map str(id) to medical concept, if(X==set() is True),unknown id
    logger.info("reading medical concept interpreter file...")
    id_med_dict = defaultdict(set)
    id_med_dict['UNK'] = 'unknown words'
    with xlrd.open_workbook(r'.\data\id_med_dict.xlsx') as data1:
        table = data1.sheets()[0]
        nrows = table.nrows
        for row_num in range(nrows):
            id_med_dict[str(row_num)] = table.row_values(row_num)[1]

    # Step 1: Read raw data.
    logger.info("reading raw data...")
    filename = r'.\data\d2bow.txt'
    cohort, vocabulary_weight, patient_num, window_size_list = read_data(filename)
    # logger.info('reading raw finished...data size : %s' % len(cohort))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    logger.info("building dictionary...")
    data, count, dictionary, reverse_dictionary = build_dataset(cohort)
    vocabulary_size = len(dictionary)
    logger.info("vocabulary_size: %s" % vocabulary_size)
    del cohort  # Hint to reduce memory.
    del vocabulary_weight
    logger.info('Most common words (+UNK) %s' % count[:10])
    # logger.info('Sample data index: {0} key: {1}'.format(data[:10], [reverse_dictionary[i] for i in data[:10]]))

    # # Step 3: Function to generate a training batch for the skip-gram model.
    # print("training a batch for the skip-gram model...")
    data_index = 0
    visit_index = 0
    # batch_size = 8
    # batch, labels = generate_batch(batch_size=batch_size, num_skips=2, skip_window=1)
    # for i in range(batch_size):
    #     # id is key's frequency order in descending order
    #     print(batch[i], reverse_dictionary[batch[i]],
    #           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    start_time = time.clock()
    # Step 4: Build and train a skip-gram model.
    logger.info("training skip-gram model...")
    batch_size = 512
    embedding_size = 100  # Dimension of the embedding vector.
    # skip_window = 8  # How many words to consider left and right.
    # num_skips = 6  # How many times to reuse an input to generate a label.
    num_sampled = 16  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by construction are also the most frequent.
    # These 3 variables are used only for displaying model accuracy, they don't affect calculation.
    valid_size = 4  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    # noinspection PyUnresolvedReferences
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_examples = np.array([dictionary['44'], dictionary['61'], dictionary['298'], dictionary['106']])

    # # get visit embedding
    # with open(path, 'rb') as pkl_file:
    #     item_vector_list = pickle.load(pkl_file, encoding='bytes')
    # logger.info("vector len %s" % len(item_vector_list))
    # init_embedding = labels = np.ndarray(shape=(len(item_vector_list), embedding_size), dtype=np.int32)
    # for index, each_list in enumerate(item_vector_list):
    #     for _ in range(embedding_size):
    #         init_embedding[index, _] = each_list[_]

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            # embeddings is the one we concern
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], 0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.5 / math.sqrt(embedding_size)))
            # nce_weights = embeddings.eval().tolist()
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.5).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    logger.info("begin training...")
    # Step 5: Begin training.
    num_steps = 100001
    # num_steps = 50001

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        logger.info('Initialized...')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                logger.info('Average loss at step {0}: {1}'.format(step, average_loss))
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    # for readable
                    valid_word = id_med_dict[valid_word]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        close_word = id_med_dict[close_word]
                        log_str = '%s %s,' % (log_str, close_word)
                    logger.info(log_str)
        final_embeddings = normalized_embeddings.eval()
        # final_embeddings.tofile("med_final_embeddings.txt")
        # logger.info('***final embeddings***')
        # logger.info('size: %s' % final_embeddings.size)
        # logger.info('item size: %s' % final_embeddings.itemsize)
        # logger.info('shape: %s' % final_embeddings.shape)
        # logger.info('ndim: %s' % final_embeddings.ndim)
        embedding_list = final_embeddings.tolist()

        logger.info("finishing iterm vector learning...")
        logger.info("learning patient embeddings...")

        med_num = 4216
        embeddings_for_file = np.ndarray(shape=(med_num, embedding_size), dtype=np.float32)
        count = 0
        for i in range(med_num):
            key = dictionary.get(str(i), 0)
            if key == 0:
                count += 1
                embeddings_for_file[i] = np.zeros(embedding_size)
            else:
                embeddings_for_file[i] = embedding_list[key]
        logger.info("count: %s" % count)
        with open(r'.\data\skip_gram_data\skip_gram_vector_{0}_{1}.pickle'.format(embedding_size, batch_size), 'wb') as vector_f:
            pickle.dump(embeddings_for_file, vector_f, protocol=2)

        logger.info("learning patient embeddings...")

        # Step 6: get patient embeddings.

        # get patient icd name for each person
        with open(".\data\d2diag.np", "rb") as fname7:
            obj1 = np.load(fname7)
            visit_info = obj1.tolist()  # separator '0', label for one visit
            patient_icd_label = list()
            patient_icd_label.append(visit_info[0])
            icd_index = 1
            for icd_index in range(1, len(visit_info)):
                if len(patient_icd_label) < patient_num:
                    if visit_info[icd_index] == 0:
                        patient_icd_label.append(visit_info[icd_index-1])
        if len(patient_icd_label) < patient_num:
            patient_icd_label.append(0)
        # logger.info(patient_icd_label)

        # icd count
        patient_icd_label_dict = defaultdict(int)
        for i in patient_icd_label:
            patient_icd_label_dict[i] += 1
        patient_icd_label_count = len(patient_icd_label_dict)

        logger.info(sorted(patient_icd_label_dict.items(), key=lambda item: -item[1])[:10])

        with open(filename, "r") as f:
            patient_num = 0
            patient_embeddings = list()

            patient_item_list = list()
            item_weight_list = list()
            lists_of_lists = list()

            line = f.readline().strip('\n')
            count_total_visit = 0
            while line:
                if count_total_visit > 1000000:
                    break
                if not line.startswith("-1"):
                    ele = line.split(',')
                    count_total_visit += 1
                    for _ in ele:
                        key = _.split(':')[0]
                        w = _.split(':')[1]
                        patient_item_list.append(key)
                        item_weight_list.append(float(w))
                        # item_weight_list.append(1)
                else:
                    # patient_icd = patient_icd_label[patient_num]  # what for?
                    patient_embedding = list()
                    item_size = len(patient_item_list)
                    for item_index, e2 in enumerate(patient_item_list):  # get medical key
                        embedding_id = dictionary[e2]  # dictionary mapping word string to embedding_id
                        patient_visit_embedding = [i for i in embedding_list[embedding_id]]
                        # if item_index > 5 or item_index < item_size-5:
                        #     patient_visit_embedding.append(embedding_list[embedding_id])
                        lists_of_lists.append(patient_visit_embedding)
                    patient_embeddings.append([sum(x) / item_size
                                               for x in zip(*lists_of_lists)])  # sum visit and divide visit_size
                    # patient_embeddings.append([sum(x)
                    #                            for x in zip(*lists_of_lists)])  # sum visit and divide visit_size
                        # lists_of_lists.append([embedding_list[dictionary[e2]] for e2 in e1]) wrong!!!!
                    # patient_embedding.append([sum(x)
                    #                            for x in zip(*lists_of_lists)])  # sum visit and divide visit_size
                    # auto_norm = tf.sqrt(tf.reduce_sum(tf.square(patient_embedding), 1, keep_dims=True))
                    # patient_embeddings.append(patient_embedding / auto_norm)

                    # temp_em = [sum(x) for x in zip(*lists_of_lists)]
                    # nom = 0
                    # for i in temp_em:
                    #     nom += i
                    # temp_em = [i/nom for i in temp_em]
                    # patient_embeddings.append(temp_em)
                    # print(diag_id, ':', [sum(x) for x in zip(*lists_of_lists)])
                    # print('*' * 6)
                    # patient_num = patient_num + 1

                    # clear list
                    patient_item_list = []
                    lists_of_lists = []
                    item_weight_list = []

                line = f.readline().strip('\n')

        logger.info("finishing patient embeddings learning...")
