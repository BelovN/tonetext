from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from convert_data import (get_w2v_convertation_data, get_tf_idf_convertation_data,
                         get_bow_convertation_data)


def get_metrics(test_y, predicted_y):
    precision = precision_score(test_y, predicted_y, average='weighted')

    recal = recall_score(test_y, predicted_y, average='weighted')
    f1 = f1_score(test_y, predicted_y, average='weighted')
    accuracy = accuracy_score(test_y, predicted_y)
    return precision, recal, f1, accuracy


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    classes = {}
    for class_index in range(model.coeff_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }

    return classes

def build_regression(train_x, train_y, test_x, test_y):
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                             multi_class='multinomial', n_jobs=-1, random_state=40)
    clf.fit(train_x, train_y)

    predicted_y = clf.predict(test_x)

    return test_y, predicted_y


def main():
    data = {
        'BAG-OF-WORDS': get_bow_convertation_data(),
        'TF-IDF': get_tf_idf_convertation_data(),
        'WORD2VEC': [get_w2v_convertation_data()],
    }

    for key, value in data.items():
        train_x, train_y, test_x, test_y = value

        test_y, predicted_y = build_regression(train_x, train_y, test_x, test_y)
        precision, recal, f1, accuracy = get_metrics(test_y, predicted_y)
        print(f'\n {key=}')
        print(f'{precision=}, {recal=}, {f1=}, {accuracy=}')


if __name__ == '__main__':
    main()
