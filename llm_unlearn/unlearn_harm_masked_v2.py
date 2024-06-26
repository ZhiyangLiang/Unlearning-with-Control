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
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)
import pdb

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

cnt = 0
attention_loss = 0.0
# def attention_mask_hook(module, inputs, outputs): # fail try
#     outputs[1][0] = torch.where(outputs[1][0] > 0.5, outputs[1][0], torch.tensor(10.0, device=outputs[1][0].device))
#     return outputs

def attention_mask_hook(module, inputs, outputs): # success try
    global attention_loss, cnt
    part_loss = torch.where(outputs[1][0] > float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()
    if cnt % 48 < 24:
        attention_loss += part_loss
    cnt += 1
    return outputs


def main(args) -> None:
    global attention_loss

    accelerator = Accelerator()
    device = accelerator.device
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_attentions=True)
    for layer in model.model.decoder.layers:  # my try
        layer.self_attn.register_forward_hook(attention_mask_hook)
    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)

    model.to(device)
    # ori_state = model.state_dict()  # my try
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load harmful data.
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )

    # Get normal data.
    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    # Load normal answer used for random mismatch.
    # normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    model.train()

    # Reference model for computing KL.
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model.to(device)

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on answer only. ############
            bad_loss = get_answer_loss("ga", bad_batch, model, device=device)
            ############ Random mismatch. ############
            # random_loss = get_rand_ans_loss(
            #     bad_batch,
            #     tokenizer,
            #     normal_ans,
            #     model,
            #     K=5,
            #     device=device,
            # )
            ############ KL on normal samples. ############
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)
            # Final loss = bad loss + random smoothing + normal loss.
            loss = (
                attention_loss / 500
                # args.bad_weight * bad_loss
                # + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )
            # Backprop.
            accelerator.backward(loss)


            # for name, param in model.named_parameters():
            #     if 'k_proj' in name or 'q_proj' in name:
            #         grad_abs = param.grad.abs()
            #         mask = grad_abs < np.percentile(grad_abs.cpu(), float(args.mask_rate))
            #         param.grad[mask] = 0
            #     elif param.grad is not None:
            #         mask = True
            #         param.grad[mask] = 0


            optimizer.step()
            # if idx % 100 == 0:
            #     print(idx)
            #     optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # if args.robust == "yes":
            #     if idx % int(args.idx) == 0:  # my try
            #         print("idx: %d" % (idx))
            #         for name, parameter in model.named_parameters():
            #             parameter.data = 0.85 * parameter.data + 0.15 * ori_state[name].data

            # Print.
            stats = (
                f"batch: {idx}, "
                # f"bad_loss: {-bad_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
                f"attention_loss / 500: {attention_loss / 500:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1
            attention_loss = 0.0

            # Save model.
            if idx % args.save_every == 0:
                model.save_pretrained(args.model_save_dir, from_pt=True)
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model.save_pretrained(args.model_save_dir, from_pt=True)
    logging.info("Unlearning finished")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", type=bool, default=False)
    # parser.add_argument("--use_lora", type=bool, default=True)

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        # default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
    )
    # parser.add_argument(
    #     "--robust",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--mask_rate",
    #     type=float,
    # )
    # parser.add_argument(
    #     "--idx",
    #     type=int,
    # )
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
