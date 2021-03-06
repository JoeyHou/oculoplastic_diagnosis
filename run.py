import os
# os.system("mkdir -p

import sys, getopt
import pandas as pd
import json
from os import listdir
import pickle

sys.path.insert(1, './src/')
from data import *
from utils import *
from trainer import *


def display_help_menu():
    print('=> [run.py: parse_arg(argv)] Error parsing command line arguments!')
    print('         -h / --helep: show help menu')
    print('         -m= / --model=: [Required] the model name. [NEED TO HAVE CORRESPONDING JSON FILE IN "./config/" !!]')
    # print('         -c / --crop: run the cropping target')
    # print('         -t / --test: running in test mode (i.e. with a toy dataset in ./data/test/)')
    # print('         -r / --run: running in training mode (i.e. with specified dataset in /data4/xuanyu/full_data_2021/')
    # print('         -s / --single-test: running single test')
    # print('         -g / --grid-search: running grid search')

def parse_arg(argv):
    try:
        opts, args = getopt.getopt(argv, "hm:cpt", ["help", "model=", "cleaning", "prep", "train"])
    except getopt.GetoptError:
        display_help_menu()
        sys.exit(2)
    # print(opts, args)
    # print(opts, args)
    target = None
    model = None
    # testing = False

    for opt, arg in opts:
        if opt == '-h':
            display_help_menu()
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-c", "--cleaning"):
            target = "data_cleaning"
        elif opt in ("-p", "--prep"):
            target = "data_prep"
        elif opt in ("-t", "--train"):
            target = "train"
    return target, model


def main():
    target, model = parse_arg(sys.argv[1:])
    # print(target, model)
    config_file_name = model + '.json'
    if config_file_name not in listdir('config'):
        print('=> [run.py: main()] No json file with name:"' + config_file_name + '" found! terminating...')
        return

    # Load json file
    with open('config/' + config_file_name) as json_file:
        config = json.load(json_file)

    T = Trainer(config)

    if target == 'data_cleaning':
        T.data_cleaning()
    elif target == 'data_prep':
        T.data_prep()
    elif target == 'train':
        T.train()



if __name__ == "__main__":
    main()
