import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Activation, SimpleRNN
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from nltk.tokenize import word_tokenize

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from convert_data import get_standartized_data, get_data


MAX_WORDS = 10000
MAX_LEN = 150


def tokenize_data(data):
    splited_data = []
    for text in data:
        splited_data.append(word_tokenize(text))
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(data)
    data_seq = tokenizer.texts_to_sequences(data)
    data_seq = sequence.pad_sequences(data_seq, maxlen=MAX_LEN)
    return data_seq


def build_first_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 100, input_length=MAX_LEN))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    learning_rate = 0.001
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def build_second_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS,
                        output_dim=50,
                        input_length=MAX_LEN))
    model.add(SimpleRNN(16))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    loss = 'sparse_categorical_crossentropy'
    metrics = ['accuracy']
    learning_rate = 0.001
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def fit_nn(train_x, train_y, model):
    verbose = 1
    epochs = 20
    batch_size = 128
    validation_split = 0.2

    history = model.fit(
        train_x,
        train_y,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split
    )

    return model, history


def check_models():
    data_x, data_y = get_standartized_data()
    data_x = tokenize_data(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)

    models = {
        'first': build_first_model(),
        # 'second': build_second_model(),
    }

    results = {
        'first': None,
        'second': None,
    }

    for key, model in models.items():
        model, history = fit_nn(train_x, train_y, model)
        model.save(key + '.model')
        results[key] = model.evaluate(test_x, test_y)
        build_results_loss(history)
        build_results_accuracy(history)



def build_results_loss(history):
    history_dict=history.history
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']

    epochs=range(1, len(history_dict['accuracy'])+1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.xlabel('Epohs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def build_results_accuracy(history):
    history_dict=history.history
    acc_values=history_dict['accuracy']
    val_acc_values=history_dict['val_accuracy']

    epochs=range(1, len(history_dict['accuracy'])+1)

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.xlabel('Epohs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    check_models()


if __name__ == '__main__':
    main()
