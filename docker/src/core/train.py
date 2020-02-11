# -*- coding: utf-8 -*-
import logging
import os
import random
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    GPT2Model,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

from util.text import TextDataset, LineByLineTextDataset
from util.checkpoint import rotate_checkpoints


# use GPT-2 as the generator
PRETRAINED_WEIGHTS = "gpt2"
# set random seed for reproducibility
SEED = 42


logger = logging.getLogger(__name__)


"""
Built on top of HuggingFace, in particular the script here
https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py

Does not support multi-GPU training
"""


def _collate(
    tokenizer: PreTrainedTokenizer, examples: List[torch.Tensor]
) -> Any:
    return pad_sequence(
        examples, batch_first=True, padding_value=tokenizer.pad_token_id
    )


def load_and_cache_examples(
    args, tokenizer: PreTrainedTokenizer, evaluate: bool = False
):
    file_path = (
        args["<eval_data_file>"] if evaluate else args["<train_data_file>"]
    )
    if args["--line-by-line"]:
        return LineByLineTextDataset(
            tokenizer,
            args,
            file_path=file_path,
            block_size=int(args["--block_size"]),
        )
    else:
        return TextDataset(
            tokenizer,
            args,
            file_path=file_path,
            block_size=int(args["--block_size"]),
        )


def set_seed(args, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.with_gpu:
        torch.cuda.manual_seed_all(seed)


def train(
    args, model: GPT2Model, tokenizer: GPT2Tokenizer
) -> Tuple[int, float]:
    """
    Train the model

    Steps:
    1. Sample and load dataset
    2. Set up optimizer and learning schedule
    3. Train model over preset number of steps
    """
    # tb_writer = SummaryWriter()

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    train_sampler = RandomSampler(train_dataset)
    collate = partial(_collate, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
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
            "weight_decay": args.weight_decay,
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
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    # Learning policy schedule
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(
            os.path.join(args.model_name_or_path, "optimizer.pt")
        )
        and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        )
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.per_gpu_train_batch_size,
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(
        "  Gradient Accumulation steps = %d", args.gradient_accumulation_steps
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split(
                "/"
            )[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
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
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = batch, batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if (
                #     args.logging_steps > 0
                #     and global_step % args.logging_steps == 0
                # ):
                #     # Log metrics
                #     if (
                #         args.evaluate_during_training
                #     ):  # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args, model, tokenizer)
                #         for key, value in results.items():
                #             tb_writer.add_scalar(
                #                 "eval_{}".format(key), value, global_step
                #             )
                #     tb_writer.add_scalar(
                #         "lr", scheduler.get_lr()[0], global_step
                #     )
                #     tb_writer.add_scalar(
                #         "loss",
                #         (tr_loss - logging_loss) / args.logging_steps,
                #         global_step,
                #     )
                #     logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
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

                    rotate_checkpoints(args, checkpoint_prefix)

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

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
    args, model: GPT2Model, tokenizer: GPT2Tokenizer, prefix=""
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size

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

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
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
