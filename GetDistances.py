import FLAGS
import os
import Utilities
import Preprocessing as pre
import skimage.io as io
from FeatureExtraction import getFeatureVector
from scipy.spatial import distance
import time as t
import datetime


def get_feature_vectors():
    print("Starting...")
    start = t.time()
    forms_filenames = os.listdir(FLAGS.TWO_FORM_DATASET_PATH)

    writers_features_dict = {}
    writers_features_count_dict = {}
    blocks_counter = 0
    forms_counter = 0
    for form_filename in forms_filenames:
        forms_counter += 1
        writer_id, form_id = Utilities.get_writer_form_id(form_filename)
        form_image = io.imread(os.path.join(FLAGS.TWO_FORM_DATASET_PATH, form_filename))
        texture_blocks = pre.Preprocessing(form_image)
        for block in texture_blocks:
            blocks_counter += 1
            if writer_id in writers_features_count_dict:
                writers_features_dict[writer_id] += getFeatureVector(block, method=1)
                writers_features_count_dict[writer_id] += 1

            else:
                writers_features_dict[writer_id] = getFeatureVector(block, method=1)
                writers_features_count_dict[writer_id] = 1

        if forms_counter % 100 == 0:
            delta = t.time() - start
            print("Finished", forms_counter, "forms and", blocks_counter, "blocks in",datetime.timedelta(seconds=delta))

    for writer in writers_features_count_dict.keys():
        writers_features_dict[writer] = writers_features_dict[writer]/writers_features_count_dict[writer]

    delta = t.time() - start
    print("Finished all", forms_counter, "forms and", blocks_counter, "blocks in", datetime.timedelta(seconds=delta))
    print("Calculating distances and writing file...")
    pairs_counter = 0
    with open("writer-distances.txt", "w") as f:
        for writer1 in writers_features_dict.keys():
            for writer2 in writers_features_dict.keys():
                if writer1 != writer2:
                    pairs_counter += 1
                    dist = distance.euclidean(writers_features_dict[writer1], writers_features_dict[writer2])
                    f.write(str(writer1) + " " + str(writer2) + " " + str(dist) + "\n")

                    if pairs_counter % 100 == 0:
                        delta = t.time() - start
                        print("Finished", pairs_counter, "pairs in", datetime.timedelta(seconds=delta))

    delta = t.time() - start
    print("Finished", pairs_counter, "pairs in", datetime.timedelta(seconds=delta))


if __name__ == '__main__':
    get_feature_vectors()
