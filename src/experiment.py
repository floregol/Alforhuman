from src.generateColor import get_next_dataset
from src.generate_mushroom import get_mushroom_dataset
from src.db_connection import store_db, get_experiment_from_db, store_db_or_get_id, update_experiment_db_entry, TABLE_EXPERIMENT, TABLE_DATABASE, TABLE_MUSHROOM_DATABASE
import random
import time
import pickle as pk
import os
import numpy as np
from bson.binary import Binary
DYNAMIC_DATASET = ['color']


class Experiment:
    def __init__(self, session_id, dataset_type, al_type, X, y, images_path, init_labeled_size, labeled, unlabeled):
        self.session_id = session_id
        self.dataset_type = dataset_type
        self.al_type = al_type
        self.X = X  # matrix format for computer
        self.images_path = images_path  # path to png images to be displayed to user
        self.y = y  # list of labels
        # size of the label set showed at the beginning
        self.init_labeled_size = init_labeled_size
        self.labeled = labeled  # list of labeled_index
        self.labeled_size = len(self.labeled)
        self.unlabeled = unlabeled  # list of unlabeled_index
        self.test_indices = unlabeled
        # list of tuple  (index, human label pred)
        self.list_human_pred_test = []
        # list of tuple  (index, human label pred)
        self.list_human_pred_train = []
        self.test_index = 0  # keep track of which test images has been shown for server version
        self.experiment_completed = False

    def update_labeled_set(self, q):
        self.q = q
        assert q in self.unlabeled
        self.labeled_size += 1
        self.unlabeled.remove(q)
        self.labeled.append(q)
        self.test_indices = self.unlabeled

    # add prediction of human during the training phase (the feedback is given)
    def add_human_prediction(self, human_pred, q):
        self.list_human_pred_train.append((q, human_pred))

    def add_test_human_pred(self, human_pred_test, q):  # add prediction at test time
        self.list_human_pred_test.append((q, human_pred_test))

    def set_experiment_completed(self):
        self.experiment_completed = True

    def increment_test_index(self):
        self.test_index = self.test_index+1

    def get_db_entry(self):
        experiment_dict = self.__dict__
        keys_to_delete = []
        for key, val in experiment_dict.items():  # check that each entry can be put in mangodb, conversion if necessary
            if type(val).__module__ == np.__name__:  # serialize 2D array y numpy
                if self.dataset_type == 'mushroom': # we dont store the database for the mushroom exp.
                    keys_to_delete.append(key)
                else:
                    experiment_dict[key] = Binary(pk.dumps(val, protocol=2))
        for key_to_del in keys_to_delete:
            del experiment_dict[key_to_del]
        return experiment_dict

    def get_updated_dict(self):
        updated_dict = {'dataset_type': self.dataset_type, 'labeled': self.labeled, 'labeled_size': self.labeled_size, 'unlabeled': self.unlabeled, 'test_indices': self.test_indices,
                        'list_human_pred_test': self.list_human_pred_test, 'list_human_pred_train': self.list_human_pred_train, 'test_index': self.test_index, 'experiment_completed': self.experiment_completed}

        return updated_dict

    def store(self, db):

        if db:
            print('STORING DB')
            updated_dict = self.get_updated_dict()
            update_experiment_db_entry(self.session_id, updated_dict)
            print('Success')
        else:
            file_path = get_dataset_file_path(self.session_id)
            with open(file_path, "wb") as f:
                pk.dump(self, f)
        print('dataset_type', self.dataset_type)
        print('label', self.labeled)
        print('labeled_size', self.labeled_size)
        print('unlabeled', self.unlabeled)
        print('test_indices', self.test_indices)
        print('list_human_pred_test', self.list_human_pred_test)
        print('list_human_pred_train', self.list_human_pred_train)
        print('test_index', self.test_index)
        print('----------------')


class ExperimentDB(Experiment):
    def __init__(self, dict1):
        if 'dataset_type' in dict1 and dict1['dataset_type'] == 'mushroom':
            # FOR NOW IT IS JUST STORED 
            X, y, images_path = get_mushroom_dataset()
            dict1['X'] = X
            dict1['y'] = y
            self.__dict__.update(dict1)
        else:
            dict1['X'] = pk.loads(dict1['X'])
            dict1['y'] = pk.loads(dict1['y'])
            self.__dict__.update(dict1)

# Build experiment object for a session id. dataset_path unusued for now


def link_dataset_to_session(session_id, dataset_type, al_type, dataset_path, db):
    init_labeled_size = 3
    if dataset_type == 'color':

        X, y, images_path = get_next_dataset()
    elif dataset_type == 'mushroom':

        X, y, images_path = get_mushroom_dataset()
    else:
        return NotImplementedError
    dataset_size = len(images_path)
    labeled = random.sample(range(dataset_size), init_labeled_size)
    unlabeled = [i for i in range(
        dataset_size) if i not in labeled]
    experiment = Experiment(session_id=session_id, dataset_type=dataset_type, al_type=al_type, X=X, y=y, images_path=images_path,
                            init_labeled_size=init_labeled_size, labeled=labeled, unlabeled=unlabeled)

    if db:
        if dataset_type == 'color':
            database_entry = {'type': dataset_type, 'X': Binary(pk.dumps(
                X, protocol=2)), 'y': Binary(pk.dumps(y, protocol=2)), 'size': dataset_size}
            db_id = store_db(collection_name=TABLE_DATABASE,
                             dict_entry=database_entry)
            experiment_dict = experiment.get_db_entry()
            experiment_dict['db_id'] = db_id
            store_db(collection_name=TABLE_EXPERIMENT,
                     dict_entry=experiment_dict)
        elif dataset_type == 'mushroom':
            list_entries = []
            for i, x in enumerate(X):
                database_entry = {'type': dataset_type, 'size': dataset_size, 'id':i}
                x_byte =  Binary(pk.dumps( x, protocol=2))
                label_byte = int(y[i])
                database_entry['x'] = x_byte
                database_entry['y']= label_byte
                list_entries.append(database_entry)
            store_db_or_get_id(collection_name=TABLE_MUSHROOM_DATABASE,
                             dict_entries=list_entries)

            experiment_dict = experiment.get_db_entry()
            experiment_dict['db_table'] = TABLE_MUSHROOM_DATABASE
            store_db(collection_name=TABLE_EXPERIMENT,
                     dict_entry=experiment_dict)

    return experiment


# Return the dataset assigned to a particular session id


def get_experiment_of_session(session_id, db):
    if db:
        experiment_dict = get_experiment_from_db(session_id)
        experiment = ExperimentDB(experiment_dict)
        return experiment
    else:
        file_path = get_dataset_file_path(session_id)
        with open(file_path, "rb") as f:
            experiment = pk.load(f)
        return experiment


def get_dataset_file_path(session_id):
    path = os.path.join('session', str(session_id))
    return os.path.join(path, 'dataset.pkl')
