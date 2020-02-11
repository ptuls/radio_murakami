#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Sample GPT-2 model.

Usage:
    sample.py <seed_text> --num_samples=<num_samples> --max_length=<max_length>

Options:
-h --help                    Show this screen.
--num_samples=<num_samples>  Number of samples to generate [default: 1].
"""
from docopt import docopt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.inference import sample_sequence
from util.log import setup_logging

PRETRAINED_WEIGHTS = "gpt2"
FORMAT = "%(levelname)s:%(asctime)s:%(message)s"

logger = setup_logging()


def main(seed_text: str, max_length: int = 100, num_samples: int = 1) -> None:
    logger.info("loading tokenizer and model")
    # load the tokenizer: what it does is to split text into a format the model can understand
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    # load the model: this may take some time as it needs to download the pretrained weights from the Internet
    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_WEIGHTS)
    generated = sample_sequence(
        model,
        tokenizer,
        max_length,
        seed_text,
        num_samples=num_samples,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
    )

    for i in range(num_samples):
        text = tokenizer.decode(
            generated[i, 0:].tolist(),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        logger.info(text)
        logger.info("\n")


if __name__ == "__main__":
    args = docopt(__doc__, version="Sample version 0.1")
    main(
        args["<seed_text>"],
        int(args["--max_length"]),
        int(args["--num_samples"]),
    )
