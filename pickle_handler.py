"""
    utils for dealing with the existence of the classifier/regression pickle and when it should / should
    not be recalculated
"""

import os
import datetime
import pickle

CONFIGURATION = {
    "pickle_life" : 30
}


def days_between(tm1, tm2):
    """ Returns days (int?) between two datetime object"""
    return (tm2 - tm1).days


def modification_datetime(file_path):
    """ Returns a datetime object representing when the file was last modified"""
    t = os.path.getmtime(file_path)
    return datetime.datetime.fromtimestamp(t)


def get_clf_unpickled(update_fn, file_path):
    """ returns the unpickled regression, or it will perform the function given to generate it """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        if days_between(datetime.datetime.now(), modification_datetime(file_path)) >= CONFIGURATION["pickle_life"]:
          return update_fn()
        else:
            return pickle.Unpickler(open(file_path, 'rb')).load()
    else:
        return update_fn()
