import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import preprocess
import word2vec
from imdb import make_corpus, find_unique_words, make_word2ind_mapping, make_target_context_word_lists
import som


def predict_email(email_data, labels, mlp_net, skipgram, word2ind, min_num_of_words):
    num_of_prediction = 0
    num_of_correct_prediction = 0
    for i in range(len(email_data)):
        corpus = email_data[i]
        unique_words = find_unique_words(corpus)
        x_test = skipgram.get_all_word_vectors(word2ind, unique_words)

        if x_test.shape[0] > 0:
            net_act = mlp_net.predict(x_test)

            curr_result = 0

            avg_net_act = np.average(net_act)

            if avg_net_act > 0.5:
                curr_result = 1

            num_of_prediction += 1

            if curr_result == labels[i]:
                num_of_correct_prediction += 1

    print(num_of_correct_prediction/num_of_prediction)


