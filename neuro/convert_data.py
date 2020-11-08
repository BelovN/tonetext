import os
import csv
import nltk
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
    data_x = []
    data_y = []

    with open(file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for data in reader:
            if(data):
                text, index = data
                data_x.append(text)
                data_y.append(index)

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


def get_all_convertation_data():
    data_x, data_y = read_dataset(os.path.join(DATA_DIR, 'data.csv'))
    bag_of_words = covert_to_bag_of_words(data_x)
    write_data_to_file(bag_of_words)


def tokenize_and_clear(text):
    tokenized_data = word_tokenize(text)
    removed_stopwords_data = remove_stopwords(tokenized_data)
    stemmed_data = stem_words(removed_stopwords_data)
    return stemmed_data


# bag of words
def fit_emb_vectorizer(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb

# bag of words
def transform_emb_vectorizer(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.transform(data)
    return emb

# tf idf
def fit_tf_idf_vectorizer(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train

# tf idf
def transform_tf_idf_vectorizer(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.transform(data)
    return train

# word to vec
def convert_to_word2vec(data):
    word2vec_path = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin.gz")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
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
    embedings = []
    for tokenized_data in data:
        emb = get_average_word2vec(tokenized_data, vectors, generate_missing=generate_missing)
        embedings.append(emb)
    return embeddings
