#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import logging

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util.inference import sample_sequence

PRETRAINED_WEIGHTS = "gpt2"
logging.basicConfig(level=logging.INFO)


def main():
    # load the tokenizer: what it does is to split text into a format the model can understand
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    # load the model: this may take some time as it needs to download the pretrained weights from the Internet
    model = GPT2LMHeadModel.from_pretrained(PRETRAINED_WEIGHTS)
    num_samples = 1
    generated = sample_sequence(
        model,
        tokenizer,
        100,
        "Greg Roodt is ",
        num_samples=num_samples,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0)

    for i in range(num_samples):
        text = tokenizer.decode(
            generated[i, 0:].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        print(text)
        print()


if __name__ == "__main__":
    main()
