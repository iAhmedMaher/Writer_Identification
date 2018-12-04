import FLAGS
import os
import Utilities
from scipy.spatial import distance
import time as t
import datetime
import operator
import FeatureExtraction as fe
import itertools


def get_feature_vectors():
    print("Starting...")
    start = t.time()
    forms_filenames = os.listdir(FLAGS.DEFAULT_DATASET_PATH)

    writers_features_dict = {}
    writers_features_count_dict = {}
    blocks_counter = 0
    forms_counter = 0
    for form_filename in forms_filenames:
        forms_counter += 1
        writer_id, form_id = Utilities.get_writer_form_id(form_filename)
        feature_vectors = fe.get_form_feature_vectors(form_filename, method='LBP')
        for vector in feature_vectors:
            blocks_counter += 1
            if writer_id in writers_features_count_dict:
                writers_features_dict[writer_id] += vector
                writers_features_count_dict[writer_id] += 1

            else:
                writers_features_dict[writer_id] = vector
                writers_features_count_dict[writer_id] = 1

        if forms_counter % 100 == 0:
            delta = t.time() - start
            print("Finished", forms_counter, "forms and", blocks_counter, "blocks in", datetime.timedelta(seconds=delta))

    for writer in writers_features_count_dict.keys():
        writers_features_dict[writer] = writers_features_dict[writer]/writers_features_count_dict[writer]

    delta = t.time() - start
    print("Finished all", forms_counter, "forms and", blocks_counter, "blocks in", datetime.timedelta(seconds=delta))
    print("Calculating distances and writing file...")

    w1_w2_distance = []
    for pair in itertools.combinations(writers_features_dict.keys(), 2):
        writer1 = pair[0]
        writer2 = pair[1]
        dist = distance.euclidean(writers_features_dict[writer1], writers_features_dict[writer2])
        w1_w2_distance.append((writer1, writer2, dist))

    sorted_w1_w2_distance = sorted(w1_w2_distance, key=operator.itemgetter(2))
    with open(FLAGS.DISTANCES_FILENAME, 'w') as f:
        for tuple in sorted_w1_w2_distance:
            f.write(str(tuple[0]) + " " + str(tuple[1]) + " " + str(tuple[2]) + '\n')

    print('Finished')


def get_top_hardest_writers(num_of_writers):
    writers_set = set()
    with open(FLAGS.DISTANCES_FILENAME, "r") as f:
        for line in f:
            line = line.rstrip().split(' ')
            writers_set.add(line[0])
            writers_set.add(line[1])

            if len(writers_set) >= num_of_writers:
                return list(writers_set)  # ALERT num_of_writers+1 could be returned

    return list(writers_set)


if __name__ == '__main__':
    print(get_top_hardest_writers(2))
