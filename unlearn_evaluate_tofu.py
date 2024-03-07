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
from utils import create_tofu_dataloader_from_dataset_paraphrased, create_tofu_dataloader_from_dataset_perturbed
import pdb

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4_perturbed.log")

# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_woall_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_woall_4_perturbed.log")

# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_perturbed.log")

# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_idk_perturbed.log")
#
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget5_idk_perturbed.log")
#
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget10_idk_perturbed.log")

# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget1_idk_perturbed.log")
#
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget5_idk_perturbed.log")
#
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_grad_ascent_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_grad_ascent_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_grad_diff_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_grad_diff_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_KL_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_KL_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_idk_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt2.7b_tofu_forget10_idk_perturbed.log")

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

def main(args) -> None:
    log1 = logging.getLogger("log1")
    log2 = logging.getLogger("log2")
    file1_handler = logging.FileHandler(f"{args.model_name}_paraphrased.log")
    file2_handler = logging.FileHandler(f"{args.model_name}_perturbed.log")

    file1_handler.setLevel(logging.DEBUG)
    file2_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file1_handler.setFormatter(formatter)
    file2_handler.setFormatter(formatter)
    log1.addHandler(file1_handler)
    log2.addHandler(file2_handler)

    accelerator = Accelerator()
    device = accelerator.device
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, output_attentions=True)
    model = AutoModelForCausalLM.from_pretrained(f"models/{args.model_name}", output_attentions=True)  # new
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

    forget_loader_paraphrased = create_tofu_dataloader_from_dataset_paraphrased(
        "data/forget01_perturbed.json", tokenizer, batch_size=args.batch_size
    )
    forget_loader_perturbed_0 = create_tofu_dataloader_from_dataset_perturbed(
        "data/forget01_perturbed.json", tokenizer, 0, batch_size=args.batch_size
    )
    forget_loader_perturbed_1 = create_tofu_dataloader_from_dataset_perturbed(
        "data/forget01_perturbed.json", tokenizer, 1, batch_size=args.batch_size
    )
    forget_loader_perturbed_2 = create_tofu_dataloader_from_dataset_perturbed(
        "data/forget01_perturbed.json", tokenizer, 2, batch_size=args.batch_size
    )
    forget_loader_perturbed_3 = create_tofu_dataloader_from_dataset_perturbed(
        "data/forget01_perturbed.json", tokenizer, 3, batch_size=args.batch_size
    )
    forget_loader_perturbed_4 = create_tofu_dataloader_from_dataset_perturbed(
        "data/forget01_perturbed.json", tokenizer, 4, batch_size=args.batch_size
    )
    model.eval()

    idx = 0
    for forget_paraphrased_batch, forget_perturbed_batch_0, forget_perturbed_batch_1, forget_perturbed_batch_2, forget_perturbed_batch_3, forget_perturbed_batch_4 in zip(forget_loader_paraphrased, forget_loader_perturbed_0, forget_loader_perturbed_1, forget_loader_perturbed_2, forget_loader_perturbed_3, forget_loader_perturbed_4):
        forget_paraphrased_loss = compute_loss(forget_paraphrased_batch, model)
        forget_perturbed_loss_0 = compute_loss(forget_perturbed_batch_0, model)
        forget_perturbed_loss_1 = compute_loss(forget_perturbed_batch_1, model)
        forget_perturbed_loss_2 = compute_loss(forget_perturbed_batch_2, model)
        forget_perturbed_loss_3 = compute_loss(forget_perturbed_batch_3, model)
        forget_perturbed_loss_4 = compute_loss(forget_perturbed_batch_4, model)
        forget_perturbed_loss = (forget_perturbed_loss_0 + forget_perturbed_loss_1 + forget_perturbed_loss_2 + forget_perturbed_loss_3 + forget_perturbed_loss_4) / 5

        stats = (
            f"batch: {idx}, "
            f"forget_paraphrased_loss: {forget_paraphrased_loss:.2f}, "
            f"forget_perturbed_loss_0: {forget_perturbed_loss_0:.2f}, "
            f"forget_perturbed_loss_1: {forget_perturbed_loss_1:.2f}, "
            f"forget_perturbed_loss_2: {forget_perturbed_loss_2:.2f}, "
            f"forget_perturbed_loss_3: {forget_perturbed_loss_3:.2f}, "
            f"forget_perturbed_loss_4: {forget_perturbed_loss_4:.2f}, "
            f"forget_perturbed_loss: {forget_perturbed_loss:.2f}, "
        )
        logging.info(stats)
        log1.info(forget_paraphrased_loss.item())
        log2.info(forget_perturbed_loss.item())
        print(stats)
        idx += 1

    logging.info("Evaluating finished")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4_5_0.8",  # 0.9188052214121167,
        # default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4",  # 0.9188052214121167, 0.6552168853235508  (1)
        # default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4",  # 0.7659314523482239, 0.615445615040676  (1)
        # default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4",  # 0.09707484379785862, 0.0  (1)
        # default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4",  # 0.0012708143485281624, 0.0  (1)

        # default="models/finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4",  # 0.0005039436209702519, 0.6436451163536847  (2)
        # default="models/finetune_opt1.3b_tofu_forget1_ga_150_maintain_4",  # 0.0012708143485281624, 0.6412973952829294  (2)
        # default="models/finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4",  # 6.608193292756245e-06, 0.0  (2)
        # default="models/finetune_opt1.3b_tofu_forget1_ga_150_woall_4",  # 1.8880552265017844e-06, 0.0  (2)

        # default="models/finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_5_0.8",  # 0.5786001416508443,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_200_onlyx_maintain_robust_cur_4_5_0.8",  # 0.2656871402817289,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_250_onlyx_maintain_robust_cur_4_5_0.8",  # 0.404587405685253,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_300_onlyx_maintain_robust_cur_4_5_0.8",  # 0.9900193288833089,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_350_onlyx_maintain_robust_cur_4_5_0.8",  # 0.9188052214121167,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_400_onlyx_maintain_robust_cur_4_5_0.8",  # 0.7659314523482239,
        # default="models/finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4",  # 0.404587405685253, 0.7026056581474868
        # default="models/finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450",  # 0.5786001416508443, 0.6951160763572563

        # default="models/finetune_opt1.3b_tofu_forget10_attn_800_onlyx_maintain_robust_cur_4_5_0.8",  # 0.7659314523482239,
        # default="models/finetune_opt1.3b_tofu_forget10_attn_850_onlyx_maintain_robust_cur_4_5_0.8",  # 0.404587405685253,
        # default="models/finetune_opt1.3b_tofu_forget10_attn_950_onlyx_maintain_robust_cur_4_5_0.8",  # 0.2656871402817289,
        # default="models/finetune_opt1.3b_tofu_forget10_attn_1000_onlyx_maintain_robust_cur_4_5_0.8",  # 0.7659314523482239,
        # default="models/finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4",  # 0.09707484379785862, 0.7034178384916713
        # default="models/finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900",  # 0.2656871402817289, 0.7121276883209432

        # default="models/finetune_opt1.3b_tofu_forget1_grad_ascent",  # 6.608193292756245e-06, 0.0
        # default="models/finetune_opt1.3b_tofu_forget1_grad_diff",  # 6.608193292756245e-06, 0.6760089235913289
        # default="models/finetune_opt1.3b_tofu_forget1_KL",  # 0.0012708143485281624, 0.6441145305328797
        # default="models/finetune_opt1.3b_tofu_forget1_idk",  # 0.09707484379785862, 0.7151656804466725
        # default="models/finetune_opt1.3b_tofu_forget5_grad_ascent",  # 1.8880552265017844e-06, 0.0
        # default="models/finetune_opt1.3b_tofu_forget5_grad_diff",  # 2.156357811320459e-05, 0.6786529649344625
        # default="models/finetune_opt1.3b_tofu_forget5_KL",  # 6.608193292756245e-06, 0.6592851238035387
        # default="models/finetune_opt1.3b_tofu_forget5_idk",  # 0.16497269950224194, 0.715414650755837
        # default="models/finetune_opt1.3b_tofu_forget10_grad_ascent",  # 6.5768913245274e-05, 0.0
        # default="models/finetune_opt1.3b_tofu_forget10_grad_diff",  # 2.156357811320459e-05, 0.6815163041809541
        # default="models/finetune_opt1.3b_tofu_forget10_KL",  # 1.8880552265017844e-06, 0.6863643883641558
        # default="models/finetune_opt1.3b_tofu_forget10_idk",  # 0.16497269950224194, 0.7007893659567589

        # new
        # default="finetune_opt2.7b_tofu_forget1_grad_ascent",  # 2.156357811320459e-05,
        # default="finetune_opt2.7b_tofu_forget1_grad_diff",  # 0.0005039436209702519,
        # default="finetune_opt2.7b_tofu_forget1_KL",  # 0.0005039436209702519,
        # default="finetune_opt2.7b_tofu_forget1_idk",  # 0.006760732303569208,

        # default="finetune_opt2.7b_tofu_forget5_grad_ascent",  # 2.156357811320459e-05,
        # default="finetune_opt2.7b_tofu_forget5_grad_diff",  # 0.00018791118070078278,
        # default="finetune_opt2.7b_tofu_forget5_KL",  # 6.5768913245274e-05,
        # default="finetune_opt2.7b_tofu_forget5_idk",  # 0.054141077480362725,

        # default="finetune_opt2.7b_tofu_forget10_grad_ascent",  # 1.8880552265017844e-06,
        # default="finetune_opt2.7b_tofu_forget10_grad_diff",  # 6.5768913245274e-05,
        # default="finetune_opt2.7b_tofu_forget10_KL",  # 1.8880552265017844e-06,
        # default="finetune_opt2.7b_tofu_forget10_idk",  # 0.054141077480362725,
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
