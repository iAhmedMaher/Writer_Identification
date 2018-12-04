import numpy as np
import model
import Utilities
import insights_extractor
import itertools
from sklearn.neighbors import KNeighborsClassifier


def run_different_experiments(train_forms_filenames, test_forms_filenames, settings):
    predictions = []
    confidences = []
    for row in settings:
        clf = row[0]
        if type(clf) == str:
            clf = Utilities.map_str_to_clf(clf)

        for feature_option in row[1]:
            prediction, confidence = model.run_trial(train_forms_filenames, test_forms_filenames, clf, feature_option)
            predictions.append(prediction)
            confidences.append(confidence)

    return predictions, confidences


def get_train_test_from_writer_id(wid_str, max_train_forms):
    related_forms = Utilities.writer_id_to_form_filenames(wid_str)
    if len(related_forms) <= max_train_forms:
        return related_forms[:-1], related_forms[-1]
    else:
        return related_forms[:max_train_forms], related_forms[-1]  # ALERT


def get_train_test_from_writers_list(wids_str_list, max_train_forms):
    train_forms = []
    test_forms = []
    for wid in wids_str_list:
        w_train_forms, w_test_form = get_train_test_from_writer_id(wid, max_train_forms)
        train_forms.append(w_train_forms)
        test_forms.append(w_test_form)

    return train_forms, test_forms


def get_train_test_iterations(num_of_iterations, mode, num_writers_per_iteration, max_train_forms):
    if mode == 'hardest':
        # TODO wrong argument to get top hardest writers
        wids_ncr = itertools.combinations(
            insights_extractor.get_top_hardest_writers(np.power(num_of_iterations, num_writers_per_iteration)),
            num_writers_per_iteration)
        batches = []
        for w_combination in wids_ncr:
            batches.append(get_train_test_from_writers_list(w_combination, max_train_forms))
            if len(batches) == num_of_iterations:
                return batches

    else:
        raise NotImplementedError("No iteration builder mode called:", mode)


def print_performance_stats(settings,
                            num_of_iterations=100,
                            mode='hardest',
                            num_writers_per_iteration=3,
                            max_train_forms=2):
    batches = get_train_test_iterations(num_of_iterations, mode, num_writers_per_iteration, max_train_forms)

    result = None
    for batch in batches:
        predictions, confidences = run_different_experiments(batch[0], batch[1], settings)
        predictions = np.array(predictions).T
        confidences = np.array(confidences).T
        true_labels = np.array(range(num_writers_per_iteration)).reshape((-1, 1))

        if result is None:
            result = predictions == true_labels
        else:
            result = np.concatenate([result, predictions == true_labels], axis=0)

    accuracies = 100 * np.count_nonzero(result, axis=0) / len(result)
    print("Accuracies", accuracies)


if __name__ == '__main__':
    '''
    As you can see, you can test multiple classifiers (str or object) with multiple feature options,
    this will run SVC+LBP, SVC+LPQ, & kNN+LBQ. If we want SVC+(LBP;LQP) (i.e concatenated), we would simply make an
    option in FeatureExtraction to return concatenated feature vectors and the caching would still work. For knowing
    correct string of classifier/string check Utilities/FeatureExtraction file respectively. The other thing you need
    to configure is FLAGS file to match your needs and paths. May the odds ever be in your favor.
    
    Notice, if it will train on w1, w2, w3, it will also test on w1, w2, w3.
    '''
    print_performance_stats([['SVC', ['LBP', 'LPQ']],
                             [KNeighborsClassifier(n_neighbors=1), ['LBP', 'LPQ']]],
                            mode='hardest',
                            num_of_iterations=100)
