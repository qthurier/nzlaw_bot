from rasa_core.policies import Policy
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core import utils
import numpy as np

actions = {'utter_greet': 0,
           'utter_goodbye': 1,
           'utter_how_can_i_help': 2,
           'utter_default': 3,
           'utter_invite_to_ask_question': 4,
           'utter_suggest_topic': 5,
           'utter_ask_if_helpful': 6,
           'action_search': 7,
           'action_answer': 8,
           'action_alternative': 9,
           'custom_action_restart': 10}

class SimplePolicy(Policy):
    def predict_action_probabilities(self, tracker, domain):

        responses = {'greet': actions['utter_how_can_i_help'],
                     'affirm': actions['utter_goodbye'],
                     'deny': actions['action_alternative'],
                     'accept_topic': actions['utter_invite_to_ask_question'],
                     'accept_question': actions['action_answer'],
                     'ask': actions['action_search'] if tracker.get_slot('topic') is not None else actions['utter_suggest_topic'],
                     'thankyou': actions['utter_goodbye'],
                     'start': actions['utter_greet'], 
                     'goodbye': actions['utter_goodbye']}

        if tracker.latest_action_name == ACTION_LISTEN_NAME: # coming from listen state
            key = tracker.latest_message.intent['name']
            action = responses[key] + 2 if key in responses else actions['utter_default']
            return utils.one_hot(action, domain.num_actions)
        elif tracker.latest_action_name == 'utter_goodbye':
            return utils.one_hot(actions['custom_action_restart'] + 2, domain.num_actions)
        elif tracker.latest_action_name == 'action_search' and tracker.get_slot('confidence') == 'high':
            return utils.one_hot(actions['action_answer'] + 2, domain.num_actions)
        elif tracker.latest_action_name == 'action_answer':
            return utils.one_hot(actions['utter_ask_if_helpful'] + 2, domain.num_actions)
        else: # listen
            return np.zeros(domain.num_actions)





