#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.inference import QuoteGenerator

PRETRAINED_WEIGHTS = "gpt2"
logging.basicConfig(level=logging.INFO)


def main(
    model_dir: str, seed_text: str, max_length: int, num_samples: int
) -> None:
    logging.info("checking for gpus")
    if torch.cuda.is_available():
        logging.info(
            f"number of CUDA enabled devices: {torch.cuda.device_count()}"
        )

        device_name = torch.cuda.get_device_name()
        logging.info(f"device name: {device_name}")

        logging.info(
            f"device CUDA compute capability: {torch.cuda.get_device_capability(0)}"
        )
        logging.info("setting device to default")
        torch.device("cuda", 0)
    else:
        logging.info("no GPU detected, defaulting to CPU")
        torch.device("cpu")

    # load the tokenizer: what it does is to split text into a format the model can understand
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    # load the model: this may take some time as it needs to download the pretrained weights from the Internet
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    generator = QuoteGenerator(
        model,
        tokenizer,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
    )
    generated_text = generator.generate(seed_text, max_length, num_samples)

    for text in generated_text:
        print(text)
        print("*----------------------------------------------------*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_text", type=str)
    parser.add_argument(
        "--max-length", dest="max_length", type=int, default=50
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=10
    )
    parser.add_argument(
        "--model-dir", dest="model_dir", type=str, default="./murakami_bot2/"
    )
    args = parser.parse_args()
    main(args.model_dir, args.seed_text, args.max_length, args.num_samples)
