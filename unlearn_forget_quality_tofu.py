import pdb
import torch
import numpy as np
from scipy.stats import ks_2samp

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
with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_paraphrased.log", "r") as file:
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
with open("finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900_perturbed.log", "r") as file:
    for line in file:
        unlearn_forget1_perturbed_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_paraphrased.log", "r") as file:
# with open("retrain95_opt1.3b_tofu_forget5_paraphrased.log", "r") as file:
with open("retrain90_opt1.3b_tofu_forget10_paraphrased.log", "r") as file:
    for line in file:
        retrain_forget1_paraphrased_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_perturbed.log", "r") as file:
# with open("retrain95_opt1.3b_tofu_forget5_perturbed.log", "r") as file:
with open("retrain90_opt1.3b_tofu_forget10_perturbed.log", "r") as file:
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
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue,
            'KS Test Forget': test_res.statistic}

# get_forget_quality(unlearn_forget_values, unlearn_perturbed_values, retrain_forget_values, retrain_perturbed_values)
get_forget_quality(np.array(unlearn_forget1_paraphrased_loss), np.array(unlearn_forget1_perturbed_loss), np.array(retrain_forget1_paraphrased_loss), np.array(retrain_forget1_perturbed_loss))

