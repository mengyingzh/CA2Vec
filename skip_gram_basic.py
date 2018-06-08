"""Revised basic word2vec example for clinical activity vector representation learning."""

# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xlrd
import pickle
import math
import random
import time
import numpy as np
import collections
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from collections import defaultdict
from collections import Counter

import logging
logger = logging.getLogger("my_skip_gram_basic")


def read_data(filename, dict_label):
    """Extract the first file as a list of words."""
    item_sub = []
    item_dosage_sub = []
    item_count_per_visit = [0]
    patient_num = 0
    count_total_visit = 0

    with open(filename, "r") as f:
        lines = f.readlines()
        count_line_max = len(lines)
        # when to stop getting data, constraint count_line or patient number
        count_line = count_line_max // 10
        if count_line <= count_line_max:
            for i in range(count_line):
                visit = lines[i].strip('\n')
                if not visit.startswith("-1"):
                    count_total_visit += 1
                    ele = visit.split(',')
                    if len(item_count_per_visit) == 0:
                        item_count_per_visit.append(len(ele))
                    else:
                        item_count_per_visit.append(item_count_per_visit[-1] + len(ele))
                    ele1 = sorted(ele, key=lambda x: dict_label[str(x.split(':')[0])])
                    # logger.info("ele %s" % ele)
                    # logger.info("ele1 %s" % ele1)
                    for _ in ele1:
                        key = _.split(':')[0]
                        key_dosage = _.split(':')[1]
                        item_sub.append(key)
                        item_dosage_sub.append(key_dosage)
                else:
                    patient_num += 1

    logger.info('total visit record number : %s' % count_total_visit)
    logger.info('total patient number : %s' % patient_num)
    # logger.info('item_number_per_visit: %s', item_number_per_visit[10:])
    # logger.info("item_number_per_visit : %s" % item_number_per_visit)
    # logger.info('total visit record number : %s' % count_total_visit)
    # logger.info('total patient number : %s' % patient_num)
    # logger.info('item_number_per_visit: %s', item_number_per_visit[10:])
    # logger.info(item_sub)
    return item_sub, item_dosage_sub, patient_num, item_count_per_visit


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]  # map of words(strings) to count of occurrences
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = defaultdict(int)
    for word, _ in count:
        dictionary[word] = len(dictionary)  # word, index: map of words(strings) to their codes(integers)

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
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # index, word

    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    global visit_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    if data_index + span > len(data):
        data_index = 0

    # buffer = collections.deque(maxlen=span)
    # buffer.extend(data[data_index:data_index + span])
    data_index += span
    # if data_index + span > item_count_per_visit[visit_index]:
    #     visit_index += 1

    for i in range(batch_size // num_skips):
        center_word_index = data_index - skip_window
        if center_word_index < item_count_per_visit[visit_index]:  # center word include in current visit
            left_span = data_index - span
            right_span = min(data_index, item_count_per_visit[visit_index])
            if left_span >= right_span:
                logger.info("******")
        else:
            left_span = item_count_per_visit[visit_index]
            right_span = data_index
            visit_index += 1
            if left_span >= right_span:
                logger.info("error**")
        buffer = collections.deque(maxlen=span)
        buffer.extend(data[left_span:right_span+1])

        context_words = [w for w in range(left_span, right_span) if w != center_word_index-left_span]
        temp_len = len(context_words)
        for c in context_words:
            if len(context_words) > num_skips:
                break
            context_words.append(c)
        words_to_use = random.sample(context_words, num_skips)
        # logger.info("context_words %s" % context_words)
        # logger.info("buffer %s" % len(buffer))
        # logger.info("words_to_use %s" % words_to_use)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[center_word_index-left_span]
            labels[i * num_skips + j, 0] = buffer[context_word % temp_len]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            # buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    visit_index = visit_index % len(item_count_per_visit)
    return batch, labels

def save_skip_gram_vector(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")

    # start program timing
    # logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    start_time = time.clock()

    # Step 0: Map str(id) to medical concept, if(X==set() is True),unknown id
    id_med_dict = dict()
    id_med_dict['UNK'] = 'unknown words'
    dict_label = dict()
    with xlrd.open_workbook(r'.\data\id_med_dict_withLABEL.xlsx') as data1:
        table = data1.sheets()[0]
        nrows = table.nrows
        for row_num in range(nrows):
            id_med_dict[str(row_num)] = table.row_values(row_num)[1]
            dict_label[str(int(table.row_values(row_num)[0]))] = row_num

    # Step 1: Read raw data.
    logger.info("reading raw data...")
    filename = r'.\data\d2bow.txt'
    vocabulary, vocabulary_weight, patient_num, item_count_per_visit = read_data(filename, dict_label)
    logger.info('reading raw finished...data size : %s' % len(vocabulary))

    # Step 2: Build the dictionary and replace rare words with UNK token.
    logger.info("building dictionary...")
    vocabulary_size = len(id_med_dict)
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    logger.info("data size: %s" % len(data))
    del vocabulary  # Hint to reduce memory.
    del vocabulary_weight
    # del item_count_per_visit
    logger.info('Most common words (+UNK) %s' % count[:15])
    logger.info('Sample data index: {0} key: {1}'.format(data[:10], [reverse_dictionary[i] for i in data[:10]]))
    vocabulary_size = min(vocabulary_size, len(dictionary))
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

    # Step 4: Build and train a skip-gram model.
    logger.info("training skip-gram model...")
    batch_size = 512
    embedding_size = 100  # Dimension of the embedding vector.
    skip_window = 2  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 32  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 8  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    # noinspection PyUnresolvedReferences
    # valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # logger.info()
    valid_examples = np.array([100, 101, 102, 103, 104, 105, 206, 307, 408])

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], 0, 1.0))
            # embed = tf.nn.embedding_lookup(tf.nn.relu(embeddings), train_inputs)
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
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
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

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
    num_steps = 100000
    # num_steps = 4000000

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        logger.info('Initialized...')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
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
            if step % 1000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    # for readable
                    valid_word = id_med_dict.get(valid_word)
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        close_word = id_med_dict.get(close_word)
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
        logger.info("dictionary %s" % len(dictionary))
        logger.info("embedding_list %s" % len(embedding_list))
    # # reorder item embedding
    #     item_embedding_list = list()
    #     for i in range(len(dictionary)):
    #         item_embedding_list.append(embedding_list[dictionary[i]])
    #     del embedding_list
    # # save patient item embedding in dictionary order:
    #     save_vector_path = r"D:\Pycharmworkspace\word2vec\data\skip_gram_data\vector.pickle"
    #     save_skip_gram_vector(item_embedding_list, save_vector_path)

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
                item_size = len(patient_item_list)
                # logger.info("item_size %s" % patient_item_list)
                for item_index, e2 in enumerate(patient_item_list):  # get medical key
                    embedding_id = dictionary[e2]  # dictionary mapping word string to embedding_id
                    patient_visit_embedding = embedding_list[embedding_id]
                    lists_of_lists.append(patient_visit_embedding)
                patient_embeddings.append([sum(x)/item_size
                                           for x in zip(*lists_of_lists)])  # sum visit and divide visit_size

                # temp_em = [sum(x) for x in zip(*lists_of_lists)]
                # nom = 0
                # for i in temp_em:
                #     nom += i
                # temp_em = [i/nom for i in temp_em]
                # patient_embeddings.append(temp_em)
                # patient_num = patient_num + 1

                # clear list
                patient_item_list = []
                lists_of_lists = []
                item_weight_list = []

            line = f.readline().strip('\n')

    logger.info("finishing patient embeddings learning...")

    # evaluate embedding cluster result
    n = 1000
    true_label = patient_icd_label[100:100+n]
    vocab = Counter()
    vocab.update(true_label)
    n_class = len(vocab)
    logger.info("vocab size: %s" % n_class)

    kmeans_model = KMeans(init='k-means++', n_clusters=n_class, n_init=n_class)
    kmeans_model.fit(patient_embeddings[:n])
    cluster_labels = kmeans_model.labels_
    logger.info("NMI: %s" % normalized_mutual_info_score(true_label, cluster_labels))
    logger.info("NMI: %s" % normalized_mutual_info_score(patient_icd_label[:n], cluster_labels))

    end_time = time.clock()
    print('run time:', end_time-start_time)
