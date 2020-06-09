#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.inference import QuoteGenerator

PRETRAINED_WEIGHTS = "gpt2"
logging.basicConfig(level=logging.INFO)


def main(seed_text: str, max_length: int, num_samples: int) -> None:
    logging.info("checking for gpus")
    logging.info(
        f"number of CUDA enabled devices: {torch.cuda.device_count()}"
    )

    device_name = torch.cuda.get_device_name()
    logging.info(f"device name: {device_name}")

    logging.info(
        f"device CUDA compute capability: {torch.cuda.get_device_capability(0)}"
    )
    logging.info("setting device")
    torch.device("cuda", 0)

    # load the tokenizer: what it does is to split text into a format the model can understand
    tokenizer = GPT2Tokenizer.from_pretrained("./murakami_bot/")
    # load the model: this may take some time as it needs to download the pretrained weights from the Internet
    model = GPT2LMHeadModel.from_pretrained("./murakami_bot/")
    generator = QuoteGenerator(
        model,
        tokenizer,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
    )
    generator.generate(seed_text, max_length, num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_text", type=str)
    parser.add_argument(
        "--max-length", dest="max_length", type=int, default=50
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=10
    )
    args = parser.parse_args()
    main(args.seed_text, args.max_length, args.num_samples)
