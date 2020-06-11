# -*- coding: utf-8 -*-
import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Generator


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuoteGenerator:
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        temperature: float,
        top_k: float,
        top_p: float,
        repetition_penalty: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def sample_sequence(self, context: str, max_length: int, num_samples: int):
        context = self.tokenizer.encode(context)
        context = torch.tensor(context, dtype=torch.long)
        # start each generated sentence with seed string
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

    def generate(self, context: str, max_length: int, num_samples: int = 1):
        sequences = self.sample_sequence(context, max_length, num_samples)
        for i in range(num_samples):
            text = self.tokenizer.decode(
                sequences[i, 0:].tolist(),
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            yield text
