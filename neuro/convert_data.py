import os
import csv
import nltk
import numpy as np
import gensim

from autocorrect import Speller

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from settings import DATA_DIR

nltk.download('punkt')


def read_dataset(file_name):
    data_x = list()
    data_y = list()

    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for data in reader:
            if(data):
                text, index = data
                data_x.append(text)
                data_y.append(int(index))

    return data_x, data_y


def get_data(count_texts=10000):
    data_x, data_y = read_dataset(os.path.join(DATA_DIR, 'data.csv'))
    return data_x[:count_texts], data_y[:count_texts]


def get_standartized_data():
    data_x, data_y = get_data()
    data_x = standartize_data(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y


def remove_stopwords(tokenized_data):
    for word in tokenized_data:
        if word in stopwords.words('english'):
            tokenized_data.remove(word)
        if word in punctuation:
            tokenized_data.remove(word)
    return tokenized_data


def stem_words(tokenized_data):
    stemmer = PorterStemmer()
    spell = Speller(lang='en')

    for i in range(len(tokenized_data)):
        tokenized_data[i] = stemmer.stem(tokenized_data[i])

    return tokenized_data


def clear_text(text):
    cleared_text = text.replace(r"http\S+", "")
    cleared_text = text.replace(r"http", "")
    cleared_text = text.replace(r"@\S+", "")
    cleared_text = text.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    cleared_text = text.replace(r"@", "at")
    cleared_text = text.lower()
    return cleared_text


def standartize_data(data):
    standart_data = []
    for text in data:
        standart_data.append(clear_text(text))

    return standart_data


# bag of words
def fit_emb_vectorizer(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


def get_bow_convertation_data():
    data_x, data_y = get_data()
    data_x = standartize_data(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, count_vectorizer = fit_emb_vectorizer(train_x)
    test_x = count_vectorizer.transform(test_x)

    return train_x, train_y, test_x, test_y


# tf idf
def fit_tf_idf_vectorizer(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


def get_tf_idf_convertation_data():
    data_x, data_y = get_data()
    data_x = standartize_data(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, tfidf_vectorizer = fit_tf_idf_vectorizer(train_x)
    test_x = tfidf_vectorizer.transform(test_x)

    return train_x, train_y, test_x, test_y


# word to vec
def convert_to_word2vec(data):
    word2vec_path = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin.gz")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# word to vec
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return ne.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = []
    for tokenized_data in data:
        emb = get_average_word2vec(tokenized_data, vectors, generate_missing=generate_missing)
        embeddings.append(emb)
    return embeddings


def get_w2v_convertation_data(generate_missing=False):
    w2v_path = os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin.gz')
    vectors = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    data_x, data_y = get_data()
    data_x = standartize_data(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x = get_word2vec_embeddings(vectors, train_x, generate_missing=generate_missing)
    test_x = get_word2vec_embeddings(vectors, test_x, generate_missing=generate_missing)

    return train_x, train_y, test_x, test_y
