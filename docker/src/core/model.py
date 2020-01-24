# -*- coding: utf-8 -*-
import torch
from transformers import GPT2Model, GPT2Tokenizer

PRETRAINED_WEIGHTS = "gpt2"


def main():
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_WEIGHTS)
    model = GPT2Model.from_pretrained(PRETRAINED_WEIGHTS)


if __name__ == "__main__":
    main()
