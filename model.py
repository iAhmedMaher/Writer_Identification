import FeatureExtraction as ft
import numpy as np


def get_x_y_train(train_forms_filenames, method):
    Y_train = []
    X_train = None

    for writer_index, writer_forms in enumerate(train_forms_filenames):
        label = writer_index

        for form_filename in writer_forms:
            feature_vectors = ft.get_form_feature_vectors(form_filename, methods=method)
            labels = [label] * len(feature_vectors)

            if X_train is None:
                X_train = feature_vectors
            else:
                X_train = np.concatenate([X_train, feature_vectors], axis=0)
            Y_train = Y_train + labels

    return X_train, Y_train


def get_guessed_writer(test_form, clf, feature_method, voting_method='majority'):
    X_test = ft.get_form_feature_vectors(test_form, methods=feature_method)

    if voting_method == 'majority':
        per_block_predictions = list(clf.predict(X_test))
        guessed_writer = max(set(per_block_predictions), key=per_block_predictions.count)  # ALERT on tie chooses first
        confidence = per_block_predictions.count(guessed_writer) / len(per_block_predictions)
    elif voting_method == 'confidence_sum':
        per_block_predictions = list(np.sum(clf.predict_proba(X_test), 0) / len(X_test))
        guessed_writer = np.argmax(per_block_predictions)
        confidence = per_block_predictions[guessed_writer]
    elif voting_method == 'square_confidence_sum':
        per_block_predictions = np.sum(np.power(clf.predict_proba(X_test),2), 0)
        per_block_predictions = per_block_predictions / np.sum(per_block_predictions)
        guessed_writer = np.argmax(per_block_predictions)
        confidence = per_block_predictions[guessed_writer]
    else:
        raise NotImplementedError("No such method for guessing writer was implemented:", voting_method)

    return guessed_writer, confidence


def get_predictions(test_forms_filenames, clf, feature_method, voting_method):
    predictions = []
    confidences = []
    for form_filename in test_forms_filenames:
        prediction, confidence = get_guessed_writer(form_filename, clf, feature_method, voting_method)
        predictions.append(prediction)
        confidences.append(confidence)

    return predictions, confidences


def train(train_forms_filenames, clf, feature_method):
    X_train, Y_train = get_x_y_train(train_forms_filenames, feature_method)
    clf.fit(X_train, Y_train)
    return clf


def run_trial(train_forms_filenames, test_forms_filenames, clf, feature_method, voting_method):
    clf = train(train_forms_filenames, clf, feature_method)
    return get_predictions(test_forms_filenames, clf, feature_method, voting_method)


