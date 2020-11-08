import csv
import random
import nltk
import os

from settings import (TRAIN_NEG_DIR, TRAIN_POS_DIR,
                      TEST_NEG_DIR, TEST_POS_DIR, DATA_DIR)

nltk.download('stopwords')



POSITIVE_INDEX = 1
NEGATIVE_INDEX = 0


def shuffle_data(data_x, data_y):
    ziped = list(zip(data_x, data_y))
    random.shuffle(ziped)
    unziped = list(zip(*ziped))
    data_x = list(unziped[0])
    data_y = list(unziped[1])

    return data_x, data_y


def prepare_data(pos_dir, neg_dir):
    data_positive_x = get_data(pos_dir)
    data_positive_y = [POSITIVE_INDEX]*len(data_positive_x)

    data_negative_x = get_data(neg_dir)
    data_negative_y = [NEGATIVE_INDEX]*len(data_negative_x)

    data_x, data_y = shuffle_data(data_positive_x+data_negative_x,
                                  data_positive_y+data_negative_y)

    return data_x, data_y


def prepare_dataset():
    train_x, train_y = prepare_data(TRAIN_POS_DIR, TRAIN_NEG_DIR)
    test_x, test_y = prepare_data(TEST_POS_DIR, TEST_NEG_DIR)

    return train_x+test_x, train_y + test_y


def accomulate_dataset():
    data_x, data_y = prepare_dataset()

    with open(os.path.join(DATA_DIR, 'data.csv'), 'w') as file:
        csv_writer = csv.writer(file)

        for i in range(len(data_x)):
            csv_writer.writerow([data_x[i].replace('<br />', ''), data_y[i]])


def head():
    with open(os.path.join(DATA_DIR, 'data.csv'), 'r') as file:
        i = 0
        for line in file:
            if i < 60:
                print(line)
                i+=1
            else:
                break


def read_text(file_dir):
    text = ''
    with open(file_dir) as file:
        for line in file:
            text += ' ' + line
    return text


def get_data(data_dir):
    data = []
    file_names = os.listdir(path=data_dir)
    for file_name in file_names:
        file_dir = os.path.join(data_dir, file_name)
        text = read_text(file_dir)
        data.append(text)
    return data
