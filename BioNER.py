# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 下午3:46
# @Author  : cp
# @File    : BioNer.py.py

import argparse
from BioNER.util import set_seed
set_seed(2333)
import warnings
warnings.filterwarnings("ignore")
from args import train_argparser, eval_argparser
from config_reader import process_configs
from BioNER import input_reader
from BioNER.trainer import NERTrainer


def __train(run_args):
    trainer = NERTrainer(run_args)
    trainer.train(train_path=run_args.train_path,
                  valid_path=run_args.valid_path,
                  types_path=run_args.types_path,
                  input_reader_cls=input_reader.JsonInputReader)


def _train():

    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = NERTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path,
                 types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():

    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval']")
