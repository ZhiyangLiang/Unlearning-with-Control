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
log1 = logging.getLogger("log1")
log2 = logging.getLogger("log2")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_perturbed.log")

# file1_handler = logging.FileHandler("retrain99_opt1.3b_tofu_forget1_paraphrased.log")
# file2_handler = logging.FileHandler("retrain99_opt1.3b_tofu_forget1_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_20th_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_20th_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_15th_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_15th_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_1_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_1_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_0.5_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_0.5_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_0_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_kl_0_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_l1_1_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_l1_1_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_l1_0.5_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_l1_0.5_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster100_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster100_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster50_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster50_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster10_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster10_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster5_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_faster5_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_wo_mask_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_wo_mask_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_wo_mask_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_wo_mask_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_maintain_wo_mask_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_maintain_wo_mask_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_maintain_mask_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_maintain_mask_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_new_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_new_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_new_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_new_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_new_thre0.85_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_new_thre0.85_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_new_thre0.85_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_new_thre0.85_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_onlyx_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_onlyx_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_new_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_new_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test3_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test3_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_ori_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_ori_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_k_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_k_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_q_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_q_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_v_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_v_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_out_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_out_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_all_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_all_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_kq_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_kq_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_ori_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_ori_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test3_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test3_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.65_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.65_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.35_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.35_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.15_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.15_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.90_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.90_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.95_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.95_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.05_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.05_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.10_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.10_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_middle_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_middle_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_last_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_last_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_fc_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_fc_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_part2_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_part2_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_test2_womaintain_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_test2_womaintain_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_worobust_womaintain_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_worobust_womaintain_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_test2_womaintain_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_test2_womaintain_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_worobust_womaintain_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_worobust_womaintain_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_worobust_paraphrased.log")
# file2_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_worobust_perturbed.log")


# file1_handler = logging.FileHandler("retrain95_opt1.3b_tofu_forget5_paraphrased.log")
# file2_handler = logging.FileHandler("retrain95_opt1.3b_tofu_forget5_perturbed.log")
# file1_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget5_1000_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget5_1000_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget5_2500_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget5_2500_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
# file1_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log")
# file2_handler = logging.FileHandler("forget5_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log")

# file1_handler = logging.FileHandler("retrain90_opt1.3b_tofu_forget10_paraphrased.log")
# file2_handler = logging.FileHandler("retrain90_opt1.3b_tofu_forget10_perturbed.log")
# file1_handler = logging.FileHandler("forget10_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log")
# file2_handler = logging.FileHandler("forget10_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log")
file1_handler = logging.FileHandler("forget10_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log")
file2_handler = logging.FileHandler("forget10_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log")

file1_handler.setLevel(logging.DEBUG)
file2_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
file2_handler.setFormatter(formatter)
log1.addHandler(file1_handler)
log2.addHandler(file2_handler)
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
    accelerator = Accelerator()
    device = accelerator.device
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_attentions=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

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
        # default="models/finetune_opt1.3b_tofu",
        # default="models/finetune_opt1.3b_tofu_retrain99",
        # default="models/forget1_opt1.3b_tofu_attn",
        # default="models/forget1_opt1.3b_tofu_ga_mismatch",
        # default="models/forget1_opt1.3b_tofu_attn_1",
        # default="models/forget1_opt1.3b_tofu_attn_2",
        # default="models/forget1_opt1.3b_tofu_attn_20th",
        # default="models/forget1_opt1.3b_tofu_attn_15th",
        # default="models/forget1_opt1.3b_tofu_attn_kl_1",
        # default="models/forget1_opt1.3b_tofu_attn_kl_0.5",
        # default="models/forget1_opt1.3b_tofu_attn_kl_0",
        # default="models/forget1_opt1.3b_tofu_attn_l1_1",
        # default="models/forget1_opt1.3b_tofu_attn_l1_0.5",
        # default="models/forget1_opt1.3b_tofu_attn_1_faster100",
        # default="models/forget1_opt1.3b_tofu_attn_1_faster50",
        # default="models/forget1_opt1.3b_tofu_attn_1_faster10",
        # default="models/forget1_opt1.3b_tofu_attn_1_faster5",
        # default="models/forget1_opt1.3b_tofu_ga_mismatch_maintain",
        # default="models/forget1_opt1.3b_tofu_ga_mismatch_maintain_wo_mask",
        # default="models/forget1_opt1.3b_tofu_attn_1_wo_mask",
        # default="models/forget1_opt1.3b_tofu_attn_1_maintain_wo_mask",
        # default="models/forget1_opt1.3b_tofu_attn_1_maintain_mask",
        # default="models/forget1_opt1.3b_tofu_attn_1_new",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_new",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new",
        # default="models/forget1_opt1.3b_tofu_attn_1_new_thre0.85",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_new_thre0.85",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_onlyx",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx",
        # default="models/forget1_opt1.3b_tofu_ga_mismatch_maintain_new",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test2",
        # default="models/forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test3",
        # default="models/forget1_opt1.3b_tofu_attn_1_ori",
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_k", # 0.09707484379785862
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_q", # 0.09707484379785862
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_v", # 0.2656871402817289
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_out", # 0.09707484379785862
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_all",  # 0.404587405685253 # all means k + q + v + out
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_fc_test2",  # 0.5786001416508443, 0.6717853344749504
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_kq", # 0.09707484379785862
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_ori", # 0.7659314523482239
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_worobust",  # 0.9188052214121167; 0.610773035705785
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_test2",  # 0.7659314523482239; 0.672129889979988
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_test3", # 0.9188052214121167; 0.6214825450066229
        # default="models/forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2",
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_test2",  # same
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.65_onlyx_test2",  # 0.7659314523482239, 0.6381496534875233
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.35_onlyx_test2",  # 0.16497269950224194, 0.7068844511432315
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.15_onlyx_test2",  # 0.2656871402817289, 0.7103984548851441
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.90_onlyx_test2",  # 0.7659314523482239, 0.669826127188087
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.95_onlyx_test2",  # 0.404587405685253, 0.6997695329603242
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.05_onlyx_test2",  # 0.2656871402817289, 0.7021557359553304
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.10_onlyx_test2",  # 0.16497269950224194, 0.7009046797030838
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_test2",  # 0.7659314523482239, 0.5001666148740868
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_middle_test2",  # 0.5786001416508443, 0.6784605665785508
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_last_test2",  # 0.2656871402817289, 0.6869226765315097
        # default="models/forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_part2_test2",  # 0.7659314523482239, 0.4970357270512562
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_test2_womaintain",  # 0.9188052214121167, 0
        # default="models/forget1_opt1.3b_tofu_attn_1_onlyx_worobust_womaintain",  # 0.0012708143485281624, 0
        # default="models/forget1_opt1.3b_tofu_ga_test2_womaintain",  # 0.00018791118070078278, 0
        # default="models/forget1_opt1.3b_tofu_ga_worobust_womaintain",  # 6.608193292756245e-06, 0
        # default="models/forget1_opt1.3b_tofu_ga_test2",  # 0.003018184077228396, 0.6406677407203231
        # default="models/forget1_opt1.3b_tofu_ga_worobust",  # 0.0005039436209702519, 0.5436325135872331

        # default="models/finetune_opt1.3b_tofu_retrain95",
        # default="models/forget5_opt1.3b_tofu_attn_1_onlyx_test2",  # 0.16497269950224194, 0.6423816431290402
        # default="models/forget5_opt1.3b_tofu_attn_1_onlyx_worobust",  # 0.2656871402817289, 0.6128486919161947

        # default="models/finetune_opt1.3b_tofu_retrain90",
        # default="models/forget10_opt1.3b_tofu_attn_1_onlyx_test2",  # 0.09707484379785862, 0.6368039791110734
        default="models/forget10_opt1.3b_tofu_attn_1_onlyx_worobust",  # 0.16497269950224194, 0.5987452366167936
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
