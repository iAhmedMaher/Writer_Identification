from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import FLAGS
import os
import sys


def writer_id_to_form_filenames(wid_str):
    all_forms = os.listdir(FLAGS.DEFAULT_DATASET_PATH)
    return [f for f in all_forms if wid_str == get_writer_form_id(f)[0]]


def get_writer_form_id(filename):
    i = 0
    while filename[i] != '_':
        i += 1

    return filename[:i], filename[i+1:len(filename)-4]


def get_list_of_all_writers():
    return list(set([get_writer_form_id(f)[0] for f in os.listdir(FLAGS.DEFAULT_DATASET_PATH)]))


def map_str_to_clf(clf_string):
    clfs_dict = {
        'tree'  : DecisionTreeClassifier(max_depth=1, random_state=1),
        'kNN'   : KNeighborsClassifier(n_neighbors=3),
        'SVC'   : SVC(C=1.0, kernel='linear', probability=True),
        'rnd'   : RandomClassifier(),
        'MLP'   : MLPClassifier(hidden_layer_sizes=(64, 32, 16, 4)),
        'GBC'   : GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        'ada'   : AdaBoostClassifier(KNeighborsClassifier(n_neighbors=3),
                                 n_estimators=50,
                                 learning_rate=1.5,
                                 algorithm="SAMME")
    }

    if clf_string in clfs_dict:
        return clfs_dict[clf_string]
    else:
        raise NotImplementedError("No such classifier option exists:", clf_string)


class RandomClassifier(object):
    def __init__(self, numbers_of_writers=3):
        self.next_count = 0
        self.numbers_of_writers = numbers_of_writers

    def get_next_prediction(self):
        return self.next_count

    def predict(self, X_test):
        predictions = []
        for _ in X_test:
            predictions.append(self.get_next_prediction())

        return predictions

    def fit(self, X_train, Y_train):
        pass


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
