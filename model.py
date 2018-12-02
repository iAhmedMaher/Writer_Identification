import Preprocessing as pre
import FeatureExtraction as ft
import skimage.io as io
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os


def get_x_y_train(train_forms_filenames, writer_indices=None):
    Y_train = []
    X_train = None

    for writer_index, writer_forms in enumerate(train_forms_filenames):
        label = writer_index if writer_indices is None else writer_indices[writer_index]

        for form_filename in writer_forms:
            form = io.imread(form_filename)
            feature_vectors = get_feature_vectors_from_form(form)
            labels = [label] * len(feature_vectors)

            if X_train is None:
                X_train = feature_vectors
            else:
                X_train = np.concatenate([X_train, feature_vectors])
            Y_train = Y_train + labels

    return X_train, Y_train


def get_feature_vectors_from_form(form):
    feature_vectors = None
    texture_blocks = pre.Preprocessing(form)
    for block in texture_blocks:
        # SUG maybe this is not an optimized solution
        if feature_vectors is None:
            feature_vectors = ft.getFeatureVector(block)
        else:
            feature_vectors = np.concatenate([feature_vectors, ft.getFeatureVector(block)], axis=0)

    return feature_vectors


def get_guessed_writer(test_form, clf):
    per_block_predictions = list(clf.predict(get_feature_vectors_from_form(test_form)))  # SUG may not be optimum
    guessed_writer = max(set(per_block_predictions), key=per_block_predictions.count)
    confidence = per_block_predictions.count(guessed_writer) / len(per_block_predictions)
    return guessed_writer, confidence


def get_predictions(test_forms_filenames, clf):
    predictions = []
    confidences = []
    for form_filename in test_forms_filenames:
        form = io.imread(form_filename)
        prediction, confidence = get_guessed_writer(form, clf)
        predictions.append(prediction)
        confidences.append(confidence)

    return predictions, confidences


def train(train_forms_filenames, clf):
    X_train, Y_train = get_x_y_train(train_forms_filenames)
    clf.fit(X_train, Y_train)
    return clf


def run_trial(train_forms_filenames, test_forms_filenames, clf):
    clf = train(train_forms_filenames, clf)
    return get_predictions(test_forms_filenames, clf)


def print_performance_stats(train_forms_filenames,
                            test_forms_filenames,
                            true_labels,
                            clf=KNeighborsClassifier(n_neighbors=1)):
    predictions, confidences = run_trial(train_forms_filenames, test_forms_filenames, clf)
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)

    result = predictions == true_labels
    accuracy = len(result[result]) / len(result) * 100
    print("Accuracy:", accuracy, "%")
    print("Average confidence:", np.average(confidences))
    print("Confidences:")
    print(confidences)


if __name__ == '__main__':
    train_forms = [
        [r'E:\handwritten_dataset\113_b04-162.png'],
        [r'E:\handwritten_dataset\107_b04-004.png'],
    ]
    test_forms = [r'E:\handwritten_dataset\107_b04-000.png']

    print_performance_stats(train_forms, test_forms, [1])
