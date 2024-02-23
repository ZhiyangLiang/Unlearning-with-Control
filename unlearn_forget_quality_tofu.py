import pdb
import torch
import numpy as np
from scipy.stats import ks_2samp

unlearn_forget1_paraphrased_loss = []
unlearn_forget1_perturbed_loss = []
retrain_forget1_paraphrased_loss = []
retrain_forget1_perturbed_loss = []

# with open("retrain99_opt1.3b_tofu_forget1_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_paraphrased.log", "r") as file:
with open("forget1_opt1.3b_tofu_attn_1_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_20th_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_15th_paraphrased.log", "r") as file:
    for line in file:
        unlearn_forget1_paraphrased_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_perturbed.log", "r") as file:
with open("forget1_opt1.3b_tofu_attn_1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_20th_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_15th_perturbed.log", "r") as file:
    for line in file:
        unlearn_forget1_perturbed_loss.append(float(line.strip()))

with open("retrain99_opt1.3b_tofu_forget1_paraphrased.log", "r") as file:
    for line in file:
        retrain_forget1_paraphrased_loss.append(float(line.strip()))

with open("retrain99_opt1.3b_tofu_forget1_perturbed.log", "r") as file:
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

