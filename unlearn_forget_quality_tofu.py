import pdb
import torch
import numpy as np
import argparse
from scipy.stats import ks_2samp

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the pretrained model.",
)
parser.add_argument(
    "--retrain_model_name",
    type=str,
)

args = parser.parse_args()

unlearn_forget1_paraphrased_loss = []
unlearn_forget1_perturbed_loss = []
retrain_forget1_paraphrased_loss = []
retrain_forget1_perturbed_loss = []

# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_perturbed.log")
# file1_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_paraphrased.log")
# file2_handler = logging.FileHandler("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_perturbed.log")

# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_maintain_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_woall_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_paraphrased.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_paraphrased.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget1_grad_ascent_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_grad_diff_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_KL_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_idk_paraphrased.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget5_grad_ascent_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_idk_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_grad_diff_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_KL_paraphrased.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget10_grad_ascent_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_grad_diff_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_KL_paraphrased.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_idk_paraphrased.log", "r") as file:

with open(f"{args.model_name}_paraphrased.log", "r") as file:
    for line in file:
        unlearn_forget1_paraphrased_loss.append(float(line.strip()))

# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_maintain_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_woall_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4_perturbed.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_perturbed.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget1_grad_ascent_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_grad_diff_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_KL_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget1_idk_perturbed.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget5_grad_ascent_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_idk_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_grad_diff_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget5_KL_perturbed.log", "r") as file:

# with open("finetune_opt1.3b_tofu_forget10_grad_ascent_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_grad_diff_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_KL_perturbed.log", "r") as file:
# with open("finetune_opt1.3b_tofu_forget10_idk_perturbed.log", "r") as file:

with open(f"{args.model_name}_perturbed.log", "r") as file:
    for line in file:
        unlearn_forget1_perturbed_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_paraphrased.log", "r") as file:
# with open("retrain95_opt1.3b_tofu_forget5_paraphrased.log", "r") as file:
# with open("retrain90_opt1.3b_tofu_forget10_paraphrased.log", "r") as file:

with open(f"{args.retrain_model_name}_paraphrased.log", "r") as file:
    for line in file:
        retrain_forget1_paraphrased_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_perturbed.log", "r") as file:
# with open("retrain95_opt1.3b_tofu_forget5_perturbed.log", "r") as file:
# with open("retrain90_opt1.3b_tofu_forget10_perturbed.log", "r") as file:

with open(f"{args.retrain_model_name}_perturbed.log", "r") as file:
    for line in file:
        retrain_forget1_perturbed_loss.append(float(line.strip()))

# print(f"unlearn_forget1_perturbed_loss: {unlearn_forget1_perturbed_loss}")
# print(f"unlearn_forget1_paraphrased_loss: {unlearn_forget1_paraphrased_loss}")
# print(f"retrain_forget1_perturbed_loss: {retrain_forget1_perturbed_loss}")
# print(f"retrain_forget1_paraphrased_loss: {retrain_forget1_paraphrased_loss}")

def get_forget_quality(unlearn_paraphrase_np_values, unlearn_perturbed_np_values, retain_paraphrase_np_values, retain_perturbed_np_values):
    unlearn_truth_ratio = np.exp(unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio = np.exp(retain_perturbed_np_values - retain_paraphrase_np_values)
    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    print(f"Forget Quality: {test_res.pvalue}, KS Test PVal Forget: {test_res.pvalue}, KS Test Forget: {test_res.statistic}")

    unlearn_paraphrase_values = np.exp(-1 * unlearn_paraphrase_np_values)
    unlearn_perturbed_values = np.exp(-1 * unlearn_perturbed_np_values)
    unlearn_curr_stat_1 = unlearn_perturbed_values / unlearn_paraphrase_values
    # unlearn_curr_stat_1 = unlearn_paraphrase_values / unlearn_perturbed_values
    # unlearn_paraphrased_perturb_ratio = np.mean(unlearn_curr_stat_1)  # original
    unlearn_paraphrased_perturb_ratio = np.log(np.mean(unlearn_curr_stat_1) + 1)
    print(f"unlearn_paraphrased_perturb_ratio: {unlearn_paraphrased_perturb_ratio}")

    retain_paraphrase_values = np.exp(-1 * retain_paraphrase_np_values)
    retain_perturbed_values = np.exp(-1 * retain_perturbed_np_values)
    retain_curr_stat_1 = retain_perturbed_values / retain_paraphrase_values
    retain_paraphrased_perturb_ratio = np.mean(retain_curr_stat_1)
    print(f"retain_paraphrased_perturb_ratio: {retain_paraphrased_perturb_ratio}")

# get_forget_quality(unlearn_forget_values, unlearn_perturbed_values, retrain_forget_values, retrain_perturbed_values)
get_forget_quality(np.array(unlearn_forget1_paraphrased_loss), np.array(unlearn_forget1_perturbed_loss), np.array(retrain_forget1_paraphrased_loss), np.array(retrain_forget1_perturbed_loss))

