from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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


def map_str_to_clf(clf_string):
    clfs_dict = {
        'tree'  : DecisionTreeClassifier(max_depth=1, random_state=1),
        'kNN'   : KNeighborsClassifier(n_neighbors=3),
        'SVC'   : SVC(C=1.0, kernel='linear'),
        'ada'   : AdaBoostClassifier(KNeighborsClassifier(n_neighbors=3),
                                 n_estimators=50,
                                 learning_rate=1.5,
                                 algorithm="SAMME")
    }

    if clf_string in clfs_dict:
        return clfs_dict[clf_string]
    else:
        raise NotImplementedError("No such classifer option exist:", clf_string)


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
