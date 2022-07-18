"""
Author: Mohamed Abdelaziz
Matr.Nr.: K12137202
Exercise 5
"""

import os
import json

from trainer import Trainer
from tester import Tester
from architectures import *

def main(config):
    if config['mode'] == 'train':
        model = Trainer(config['network_config'], **config['trainer_config']).train()
        # TODO:
            # Transfer Learning
    elif config['mode'] == 'test':
        results = Tester(config['network_config'], **config['tester_config']).test()
    else:
        print('modes supported:\ttrain, test')
        # TODO:
            # Inference

if __name__ == "__main__":
    import argparse
    import json

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_file", type=str, help="Path to JSON config file", default='config.json')
        args = parser.parse_args()
        with open(args.config_file) as cf:
            config = json.load(cf)
    except:
        print('Proceeding with default config')
        try:
            with open('config.json') as cf:
                config = json.load(cf)
        except FileNotFoundError:
            raise ValueError('Configuration File Not Found')
        except:
            raise NotImplementedError('config.json is ahving some problems dude!')

    if config:
        main(config)
