import time
import os


def generate_questions():
    questions_form = {'name': 'What is you name my little pumpkin?'}
    return questions_form


def create_backend_session(session_id):
    if not os.path.exists('session'):
        os.mkdir('session')
    os.mkdir(os.path.join('session', str(session_id)))


def generate_session_id():
    return time.time()  # best session id idea in the world (maybe not)


def store_form(session_id, user_form):
    # TODO store form answer whatever
    pass
