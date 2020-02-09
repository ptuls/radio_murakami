# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, filter_value=-float("Inf")
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


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
    with torch.no_grad():
        for _ in range(length):
            inputs = {"input_ids": generated}
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (
                temperature if temperature > 0 else 1.0
            )

            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1),
                    num_samples=num_samples,
                    replacement=True,
                )
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
    return generated
