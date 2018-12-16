import numpy as np
import model
import Utilities
import insights_extractor
import itertools
from sklearn.neighbors import KNeighborsClassifier
import random
import datetime


def run_different_experiments(train_forms_filenames, test_forms_filenames, settings, voting_method):
    predictions = []
    confidences = []
    for row in settings:
        clf = row[0]
        if type(clf) == str:
            clf = Utilities.map_str_to_clf(clf)

        for feature_option in row[1]:
            prediction, confidence = model.run_trial(train_forms_filenames,
                                                     test_forms_filenames,
                                                     clf,
                                                     feature_option,
                                                     voting_method)
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


def get_train_test_iterations(counter, mode, num_writers_per_iteration, max_train_forms):
    if mode == 'hardest':
        wids_ncr = itertools.combinations(
            insights_extractor.get_top_hardest_writers(counter), num_writers_per_iteration)
        batches = []
        for w_combination in wids_ncr:
            batches.append(get_train_test_from_writers_list(w_combination, max_train_forms))
        return batches

    elif mode == 'random':
        all_writers = Utilities.get_list_of_all_writers()
        sample_writers = [all_writers[i] for i in random.sample(range(len(all_writers)), counter)]
        wids_ncr = itertools.combinations(sample_writers, num_writers_per_iteration)
        batches = []
        for w_combination in wids_ncr:
            batches.append(get_train_test_from_writers_list(w_combination, max_train_forms))
        return batches

    elif mode == 'hardest_pair':
        wids_ncr = insights_extractor.get_hardest_pairs(counter)
        batches = []
        for w_combination in wids_ncr:
            batches.append(get_train_test_from_writers_list(w_combination, max_train_forms))
        return batches

    else:
        raise NotImplementedError("No iteration builder mode called:", mode)


def print_performance_stats(settings,
                            voting_method='majority',
                            counter=100,
                            mode='hardest',
                            num_writers_per_iteration=3,
                            max_train_forms=2,
                            store_wrong_classification=False):
    print("Starting test ...")
    print("Getting batches ...")
    batches = get_train_test_iterations(counter, mode, num_writers_per_iteration, max_train_forms)

    result = None
    print("Computing training and testing ...")
    wrong_classifications = []  # ALERT works only on single classifer and feature
    for i, batch in enumerate(batches):
        predictions, confidences = run_different_experiments(batch[0], batch[1], settings, voting_method=voting_method)
        predictions = np.array(predictions).T
        confidences = np.array(confidences).T
        true_labels = np.array(range(num_writers_per_iteration)).reshape((-1, 1))

        if result is None:
            result = predictions == true_labels
        else:
            result = np.concatenate([result, predictions == true_labels], axis=0)

        if store_wrong_classification:
            filter = np.array(predictions != true_labels).reshape(-1, )
            if len(np.array(batch[1])[filter]) != 0:
                filter = np.array(predictions != true_labels).reshape(-1,)
                wrong_classifications.append([
                    batch[0],
                    np.array(batch[1])[filter],
                    np.array(predictions)[filter],
                    np.array(true_labels)[filter]
                ])

        if i % 100 == 0:
            print('Current batch number:', i)

    if store_wrong_classification:
        log_filename = 'wrong_classifications' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.txt'
        with open(log_filename, 'w') as f:
            for wrongie in wrong_classifications:
                f.write('Training forms:\n')
                for i, writer_forms in enumerate(wrongie[0]):
                    f.write('Writer'+str(i)+': ')
                    for writer_form in writer_forms:
                        f.write(writer_form + ', ')

                    f.write('\n')

                f.write('Test forms wrongly classified:\n')
                for i in range(len(wrongie[1])):
                    f.write('Form '+wrongie[1][i] + ' classified as ' + str(wrongie[2][i]) + ' while it is ' + str(wrongie[3][i]) + '\n')

                f.write("***********\n")

    accuracies = 100 * np.count_nonzero(result, axis=0) / len(result)
    print("Accuracies", accuracies)


if __name__ == '__main__':
    # ALERT for store_wrong_classification use one Feature option and classifier
    print_performance_stats([['SVC', ['LPQ', 'LBP', 'LBP;LPQ']],
                             ['kNN', ['LPQ', 'LBP', 'LBP;LPQ']]],
                            mode='random',
                            num_writers_per_iteration=3,
                            counter=70,
                            store_wrong_classification=False,
                            voting_method='confidence_sum')
    print_performance_stats([['SVC', ['LPQ', 'LBP', 'LBP;LPQ']],
                             ['kNN', ['LPQ', 'LBP', 'LBP;LPQ']]],
                            mode='random',
                            num_writers_per_iteration=3,
                            counter=70,
                            store_wrong_classification=False,
                            voting_method='confidence_sum')
    print_performance_stats([['SVC', ['LPQ', 'LBP', 'LBP;LPQ']],
                             ['kNN', ['LPQ', 'LBP', 'LBP;LPQ']]],
                            mode='random',
                            num_writers_per_iteration=3,
                            counter=70,
                            store_wrong_classification=False,
                            voting_method='majority')
    print_performance_stats([['SVC', ['LPQ', 'LBP', 'LBP;LPQ']],
                             ['kNN', ['LPQ', 'LBP', 'LBP;LPQ']]],
                            mode='random',
                            num_writers_per_iteration=3,
                            counter=70,
                            store_wrong_classification=False,
                            voting_method='majority')
