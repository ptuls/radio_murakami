# -*- coding: utf-8 -*-
import logging
import os
import glob
import random
import re
import shutil

import numpy as np
from tqdm import tqdm, trange
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

from src.util.text import TextDataset, LineByLineTextDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


PRETRAINED_WEIGHTS = "gpt2"
SEED = 42


logger = logging.getLogger(__name__)


"""
Built on top of HuggingFace, in particular the script here https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
"""


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("C:\gpt2")
    model = GPT2LMHeadModel.from_pretrained("C:\gpt2")


def load_and_cache_examples(args, tokenizer: GPT2Tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer, args, file_path=file_path, block_size=args.block_size
        )


def set_seed(with_gpu):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if with_gpu:
        torch.cuda.manual_seed_all(SEED)


def _sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(
                ".*{}-([0-9]+)".format(checkpoint_prefix), path
            )
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(
        args, checkpoint_prefix, use_mtime
    )
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[
        :number_of_checkpoints_to_delete
    ]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def mask_tokens(
    inputs: torch.Tensor, tokenizer: GPT2Tokenizer, args
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token
    )

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long
    )
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(
    model_name_or_path,
    train_dataset,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
) -> Tuple[int, float]:
    """
    Train the model
    """
    tb_writer = SummaryWriter()

    max_steps = 100
    gradient_accumulation_steps = 10
    train_batch_size = 10
    max_grad_norm = 1.0

    def collate(examples: List[torch.Tensor]):
        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collate,
    )

    t_total = max_steps
    num_train_epochs = (
        max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=0.001, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=3, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size * gradient_accumulation_steps,
    )
    logger.info(
        "  Gradient Accumulation steps = %d", gradient_accumulation_steps
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if model_name_or_path and os.path.exists(model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info(
                "  Continuing training from global step %d", global_step
            )
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=False
    )
    set_seed(True)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=False
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            inputs = inputs.to(0)
            labels = labels.to(0)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar(
                        "lr", scheduler.get_lr()[0], global_step
                    )
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir,
                        "{}-{}".format(checkpoint_prefix, global_step),
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(
                        args, os.path.join(output_dir, "training_args.bin")
                    )
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s",
                        output_dir,
                    )

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
    args, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prefix=""
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (
            mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        )
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = (
                model(inputs, masked_lm_labels=labels)
                if args.mlm
                else model(inputs, labels=labels)
            )
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(
        eval_output_dir, prefix, "eval_results.txt"
    )
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


if __name__ == "__main__":
    main()
