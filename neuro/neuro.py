import csv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_WORDS = 5000


def split_data(data_x, data_y, k):
    len_data = len(data_x)
    index = math.ceil(k*len_data)

    train_x = data_x[:index]
    train_y = data_y[:index]

    test_x = data_x[index:]
    test_y = data_y[index:]

    return train_x, train_y, test_x, test_y


#
#
#
#
# model = Sequential()
# model.add(Embedding(input_dim=max_words,
#                     output_dim=50,
#                     input_length=max_len))
# model.add(SimpleRNN(8))
# model.add(Dense(1, activation='sigmoid'))
