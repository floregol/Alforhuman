
import random


def generate_next_query(session_id, dataset):
    q = random.choice(dataset.unlabeled)
    dataset.update_labeled_set(q)
    return q

    
