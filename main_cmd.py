
import random
from server_business import  *
from cmd.cmd_util import show_to_the_user, query_user
from server_business.server_business import ServerBusiness

"""
This is simulating the client experience. 
A command line version of the experiment.
"""


if __name__ == "__main__":
    dataset_type = 'color'
    NUM_TRAIN_EXAMPLES = 5
    NUM_TEST_EXAMPLES = 5
    serverBusiness = ServerBusiness(db=True) # change for local storage or use db
    print('Trying out the', dataset_type, 'dataset')
   
    al_type = random.randint(0, 2) # Flip a coin to decide if we get Active Learning or Random
    session_id, questions = serverBusiness.start_session()
    # Normally we ask the questions to fill the form, here we skip it and just return an empty form.
    serverBusiness.receive_form(session_id, user_form=None)

    # Initialize dataset, either a static or dynamic dataset. After that the session id is linked to the path of the dataset.
    serverBusiness.initialize_dataset(session_id, dataset_type, al_type)

    # get raw features (vector) of x, not the path to the images.
    X_0, y_0 = serverBusiness.get_first_images(session_id, return_raw_features=True)

    # display first images to the user
    show_to_the_user(X_0, y_0)

    # get from server first image to display
    X_query, true_y, q = serverBusiness.start_active_learning(
        session_id, return_raw_features=True)

    # get from the  user its classification
    human_label = query_user(X_query, true_y)

    keepgoing = True
    counter = 0
    while keepgoing:

        serverBusiness.store_active_learning_pred(session_id, human_label, q)# store previous answer,
        X_query, true_y, q = serverBusiness.active_learning_iteration(session_id, return_raw_features=True) # get next query
        human_label = query_user(X_query, true_y)
        counter += 1
        if counter == NUM_TRAIN_EXAMPLES:
            keepgoing = False

    
    counter = 0
    keepgoing = True
    while keepgoing:
        X_query, true_y, q = serverBusiness.test_iteration(session_id,  return_raw_features=True)
        human_label_test = query_user(X_query)
        serverBusiness.store_pred(session_id, human_label_test,q)
        counter += 1
        if counter == NUM_TEST_EXAMPLES:
            keepgoing = False


    serverBusiness.signal_end_experiment(session_id)
    print('FINISHED')
