# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuoteGenerator:
    def __init__(
        self, model, tokenizer, temperature, top_k, top_p, repetition_penalty
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample_sequence(self, context, max_length, num_samples):
        context = self.tokenizer.encode(context)
        context = torch.tensor(context, dtype=torch.long)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = self.model.generate(
            context,
            do_sample=True,
            max_length=max_length,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        return generated

    def generate(self, context, max_length, num_samples=1):
        sequences = self.sample_sequence(context, max_length, num_samples)
        for i in range(num_samples):
            text = self.tokenizer.decode(
                sequences[i, 0:].tolist(),
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            print(text)
            print("*----------------------------------------------------*")
