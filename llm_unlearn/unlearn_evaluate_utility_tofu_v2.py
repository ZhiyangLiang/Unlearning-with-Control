# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import create_tofu_dataloader_from_dataset, create_tofu_dataloader_from_dataset_paraphrased, create_tofu_dataloader_from_dataset_perturbed
import pdb

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def compute_loss(batch, model, device="cuda:0"):
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.

    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss

def eval_dataset(data_name, model, tokenizer, eval_log1, eval_log2, eval_log3):
    forget_loader_original = create_tofu_dataloader_from_dataset(
        f"data/{data_name}.json", tokenizer, batch_size=1
    )
    forget_loader_paraphrased = create_tofu_dataloader_from_dataset_paraphrased(
        f"data/{data_name}.json", tokenizer, batch_size=1
    )
    forget_loader_perturbed_0 = create_tofu_dataloader_from_dataset_perturbed(
        f"data/{data_name}.json", tokenizer, 0, batch_size=1
    )
    forget_loader_perturbed_1 = create_tofu_dataloader_from_dataset_perturbed(
        f"data/{data_name}.json", tokenizer, 1, batch_size=1
    )
    forget_loader_perturbed_2 = create_tofu_dataloader_from_dataset_perturbed(
        f"data/{data_name}.json", tokenizer, 2, batch_size=1
    )
    forget_loader_perturbed_3 = create_tofu_dataloader_from_dataset_perturbed(
        f"data/{data_name}.json", tokenizer, 3, batch_size=1
    )
    forget_loader_perturbed_4 = create_tofu_dataloader_from_dataset_perturbed(
        f"data/{data_name}.json", tokenizer, 4, batch_size=1
    )

    model.eval()

    idx = 0
    for forget_original_batch, forget_paraphrased_batch, forget_perturbed_batch_0, forget_perturbed_batch_1, forget_perturbed_batch_2, forget_perturbed_batch_3, forget_perturbed_batch_4 in zip(forget_loader_original, forget_loader_paraphrased, forget_loader_perturbed_0, forget_loader_perturbed_1, forget_loader_perturbed_2, forget_loader_perturbed_3, forget_loader_perturbed_4):
        forget_original_loss = compute_loss(forget_original_batch, model)
        forget_paraphrased_loss = compute_loss(forget_paraphrased_batch, model)
        forget_perturbed_loss_0 = compute_loss(forget_perturbed_batch_0, model)
        forget_perturbed_loss_1 = compute_loss(forget_perturbed_batch_1, model)
        forget_perturbed_loss_2 = compute_loss(forget_perturbed_batch_2, model)
        forget_perturbed_loss_3 = compute_loss(forget_perturbed_batch_3, model)
        forget_perturbed_loss_4 = compute_loss(forget_perturbed_batch_4, model)
        forget_perturbed_loss = (forget_perturbed_loss_0 + forget_perturbed_loss_1 + forget_perturbed_loss_2 + forget_perturbed_loss_3 + forget_perturbed_loss_4) / 5

        stats = (
            f"batch: {idx}, "
            f"forget_original_loss: {forget_original_loss:.2f}, "
            f"forget_paraphrased_loss: {forget_paraphrased_loss:.2f}, "
            f"forget_perturbed_loss_0: {forget_perturbed_loss_0:.2f}, "
            f"forget_perturbed_loss_1: {forget_perturbed_loss_1:.2f}, "
            f"forget_perturbed_loss_2: {forget_perturbed_loss_2:.2f}, "
            f"forget_perturbed_loss_3: {forget_perturbed_loss_3:.2f}, "
            f"forget_perturbed_loss_4: {forget_perturbed_loss_4:.2f}, "
            f"forget_perturbed_loss: {forget_perturbed_loss:.2f}, "
        )
        logging.info(f"for {data_name}:")
        logging.info(stats)
        eval_log1.info(forget_original_loss.item())
        eval_log2.info(forget_paraphrased_loss.item())
        eval_log3.info(forget_perturbed_loss.item())
        print(stats)
        idx += 1

def main(args) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_attentions=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    eval_dataset("retain_perturbed", model, tokenizer, log1, log2, log3)
    logging.info("Evaluating finished")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="models/finetune_opt1.3b_tofu_retrain99",
        # default="models/forget1_opt1.3b_tofu_attn",
        # default="models/forget1_opt1.3b_tofu_ga_mismatch",
        # default="models/forget1_opt1.3b_tofu_attn_1",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--file1",
        type=str,
    )
    parser.add_argument(
        "--file2",
        type=str,
    )
    parser.add_argument(
        "--file3",
        type=str,
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )

    log1 = logging.getLogger("log1")
    log2 = logging.getLogger("log2")
    log3 = logging.getLogger("log3")

    file1_handler = logging.FileHandler(args.file1)
    file2_handler = logging.FileHandler(args.file2)
    file3_handler = logging.FileHandler(args.file3)

    file1_handler.setLevel(logging.DEBUG)
    file2_handler.setLevel(logging.DEBUG)
    file3_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    file1_handler.setFormatter(formatter)
    file2_handler.setFormatter(formatter)
    file3_handler.setFormatter(formatter)

    log1.addHandler(file1_handler)
    log2.addHandler(file2_handler)
    log3.addHandler(file3_handler)

    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
