import Preprocessing as pre
import FeatureExtraction as ft
import skimage.io as io
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os


def get_x_y_train(train_forms_filenames, method, writer_indices=None):
    Y_train = []
    X_train = None

    for writer_index, writer_forms in enumerate(train_forms_filenames):
        label = writer_index if writer_indices is None else writer_indices[writer_index]

        for form_filename in writer_forms:
            feature_vectors = np.concatenate(ft.get_form_feature_vectors(form_filename, method=method), axis=0)
            labels = [label] * len(feature_vectors)

            if X_train is None:
                X_train = feature_vectors
            else:
                X_train = np.concatenate([X_train, feature_vectors])
            Y_train = Y_train + labels

    return X_train, Y_train


def get_guessed_writer(test_form, clf, feature_method, voting_method='majority'):
    X_test = np.concatenate(ft.get_form_feature_vectors(test_form, method=feature_method), axis=0)
    per_block_predictions = list(clf.predict(X_test))

    if voting_method == 'majority':
        guessed_writer = max(set(per_block_predictions), key=per_block_predictions.count)
        confidence = per_block_predictions.count(guessed_writer) / len(per_block_predictions)
    else:
        raise NotImplementedError("No such method for guessing writer was implemented:", voting_method)

    return guessed_writer, confidence


def get_predictions(test_forms_filenames, clf, feature_method):
    predictions = []
    confidences = []
    for form_filename in test_forms_filenames:
        prediction, confidence = get_guessed_writer(form_filename, clf, feature_method)
        predictions.append(prediction)
        confidences.append(confidence)

    return predictions, confidences


def train(train_forms_filenames, clf, feature_method):
    X_train, Y_train = get_x_y_train(train_forms_filenames, feature_method)
    clf.fit(X_train, Y_train)
    return clf


def run_trial(train_forms_filenames, test_forms_filenames, clf, feature_method='LBP'):
    clf = train(train_forms_filenames, clf, feature_method)
    return get_predictions(test_forms_filenames, clf, feature_method)


