# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_sequence(
    model,
    tokenizer,
    length,
    context,
    num_samples=1,
    temperature=0,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.0,
    device="cpu",
):
    context = tokenizer.encode(context)
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    generated = model.generate(
        generated,
        do_sample=True,
        max_length=50,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    return generated
