from session_util import generate_session_id, create_backend_session, generate_questions, store_form
from dataset import generate_initial_dataset, get_dataset_of_session
from active_learning import generate_next_query
# Server Business 1


def start_session():
    session_id = generate_session_id()
    create_backend_session(session_id)
    questions = generate_questions()
    return session_id, questions


# Server Business 2
def receive_form(session_id, user_form):
    store_form(session_id, user_form)


def initialize_dataset(session_id, dataset_type, dataset_path):
    dataset = generate_initial_dataset(session_id, dataset_type, dataset_path)
    dataset.store()


def get_first_images(session_id, return_raw_features=False):
    dataset = get_dataset_of_session(session_id)
    if return_raw_features:
        return [dataset.X[i,:] for i in dataset.labeled], dataset.y[dataset.labeled]
    else:
        return [dataset.images_path[i] for i in dataset.labeled], dataset.y[dataset.labeled]
# Server Business 3


def start_active_learning(session_id, return_raw_features=False):
    dataset = get_dataset_of_session(session_id)
    q = generate_next_query(session_id, dataset)
    dataset.store()
    if return_raw_features:
        return dataset.X[q,:], dataset.y[q], q
    else:
        return dataset.images_path[q], dataset.y[q], q

# Server Business 4


def active_learning_iteration(session_id, human_label: int, q: int, return_raw_features=False):
    dataset = get_dataset_of_session(session_id)
    dataset.add_human_prediction(human_label, q)
    q = generate_next_query(session_id, dataset)
    dataset.store()
    if return_raw_features:
        return dataset.X[q,:], dataset.y[q], q
    else:
        return dataset.images_path[q], dataset.y[q], q

# Server Business 5

def test_time(session_id, return_raw_features=False):
    dataset = get_dataset_of_session(session_id)
    
    if return_raw_features:
        return [dataset.X[i,:] for i in dataset.test_indices], dataset.y[dataset.test_indices]
    else:
        return [dataset.images_path[i] for i in dataset.test_indices], dataset.y[dataset.test_indices]

# Server Business 6


def get_user_prediction(session_id, human_labels: list, test_indices: list):
    dataset = get_dataset_of_session(session_id)
    store_predited_label(session_id, human_label, q)
    return 'Goodbye Thank you'
