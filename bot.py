from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core import utils
from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.events import SlotSet
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.channels import HttpInputChannel

logger = logging.getLogger(__name__)

from channels import *
from retrieve import *
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz, process

topic_lookup = {'acc': 18, 'neighborhood life': 10, 'relationships and break-ups': 24}

class ActionAlternative(Action):
    def name(self):
        return 'action_alternative'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("ok let's see..")
        best_guess_question_list = tracker.get_slot('best_guess_question_list')
        button_list = [{'title': question[:100], 
                        'payload': '/accept_question{"suggestion": ' + str(i) + '}'} for i, question in enumerate(best_guess_question_list)]
        dispatcher.utter_button_message('here are some alternatives, which of those best match with your question?', button_list)
        return []

class ActionAnswer(Action):
    def name(self):
        return 'action_answer'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("here's what I found:")
        best_guess_answer_list = tracker.get_slot('best_guess_answer_list')
        accepted_suggestion = tracker.get_slot('suggestion')
        answer = best_guess_answer_list[accepted_suggestion]
        dispatcher.utter_message(BeautifulSoup(answer, 'lxml').text)
        return []

class ActionSearch(Action):
    def name(self):
        return 'action_search'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message('I''m on it..')
        question_asked = tracker.latest_message.text
        topic = tracker.get_slot('topic')
        best_guess_df = get_similar_questions_answers(question_asked, topic=topic_lookup[topic])
        best_guess_question_list = best_guess_df.question.tolist()
        best_guess_answer_list = best_guess_df.answer.tolist()
        if best_guess_df['score'].iloc[0] >= 0.75:
            return [SlotSet("best_guess_question_list", best_guess_question_list), 
                    SlotSet("best_guess_answer_list", best_guess_answer_list),
                    SlotSet("suggestion", 0), 
                    SlotSet("confidence", "high")]
        else: 
            button_list = [{'title': question[:100], 
                            'payload': '/accept_question{"suggestion": ' + str(i) + '}'} for i, question in enumerate(best_guess_question_list)]
            dispatcher.utter_button_message('humm I''m not quite sure.. which of those best match with your question?', button_list)
            return [SlotSet("best_guess_question_list", best_guess_question_list), 
                    SlotSet("best_guess_answer_list", best_guess_answer_list),
                    SlotSet("confidence", "low")]


def train_dialogue(domain_file="domain.yml",
                   model_path="models/dialogue",
                   training_data_file="data/stories.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()])

    agent.train(
            training_data_file,
            max_history=3,
            epochs=400,
            batch_size=100,
            validation_split=0.0
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data('data/training.json')
    trainer = Trainer(RasaNLUConfig("nlu_model_config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist('models/nlu/', fixed_model_name="current")

    return model_directory


def run(serve_forever=True):
    interpreter = RasaNLUInterpreter("models/nlu/default/current")
    agent = Agent.load("models/dialogue", interpreter=interpreter)
    if serve_forever:
        #agent.handle_channel(ConsoleInputChannel())
        logger.info("foo")
        agent.handle_channel(HttpInputChannel(3000, "/app", nzlaw_bot))
        logger.info("bar")

    return agent


def run_online(input_channel, 
               domain_file="domain.yml",
               training_data_file='data/stories.md'):
    interpreter = RasaNLUInterpreter("models/nlu/default/current")
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(
            description='starts the bot')

    parser.add_argument(
            'task',
            choices=["train-nlu", "train-dialogue", "run-online", "run"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "run":
        run()
    elif task == "run-online":
        run_online(ConsoleInputChannel())
    else:
        warnings.warn("Need to pass either 'train-nlu', 'train-dialogue', 'run-online' or "
                      "'run' to use the script.")
        exit(1)