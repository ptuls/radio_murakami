#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.inference import sample_sequence

PRETRAINED_WEIGHTS = "gpt2"
logging.basicConfig(level=logging.INFO)


def main(seed_text: str):
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

    num_samples = 100
    generated = sample_sequence(
        model,
        tokenizer,
        100,
        seed_text,
        num_samples=num_samples,
        temperature=0.8,
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
        print(text)
        print("*----------------------------------------------------*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_str", type=str)
    args = parser.parse_args()
    main(args.seed_str)
