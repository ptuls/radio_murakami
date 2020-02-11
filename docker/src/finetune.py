#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Finetune GPT-2 model.

Usage:
    finetune.py <train_data_file> <eval_data_file> [--line-by-line] --block_size=<block_size>

Options:
-h --help                    Show this screen.
--line-by-line               Read text file line by line (no caching)
--block-size                 Block size of text [default: 512]
"""
from core.train import train
from docopt import docopt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.log import setup_logging

PRETRAINED_WEIGHTS = "gpt2"
logger = setup_logging()


def main(args) -> None:
    logger.info("loading tokenizer and model")
    # load the tokenizer: what it does is to split text into a format the model can understand
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    # load the model: this may take some time as it needs to download the pretrained weights from the Internet
    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_WEIGHTS)
    train(args, model, tokenizer)


if __name__ == "__main__":
    args = docopt(__doc__, version="0.1")
    main(args)
