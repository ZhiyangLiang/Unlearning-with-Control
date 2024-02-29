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
# with open("forget1_opt1.3b_tofu_attn_1_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_20th_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_15th_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_kl_1_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_kl_0.5_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_l1_1_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_l1_0.5_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster100_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster50_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster10_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster5_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_wo_mask_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_wo_mask_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_maintain_wo_mask_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_maintain_mask_paraphrased.log", "r") as file:
# # with open("forget1_opt1.3b_tofu_attn_1_new_paraphrased.log", "r") as file:
# # with open("forget1_opt1.3b_tofu_attn_1_mask_new_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_new_thre0.85_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_new_thre0.85_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_onlyx_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_new_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test3_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_ori_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_k_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_q_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_v_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_out_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_all_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_kq_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_ori_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test3_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.65_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.35_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.15_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.90_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.95_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.05_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.10_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_middle_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_last_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_fc_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_part2_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_test2_womaintain_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_worobust_womaintain_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_test2_womaintain_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_worobust_womaintain_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_test2_paraphrased.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_worobust_paraphrased.log", "r") as file:

# with open("forget5_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget5_1000_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget5_2500_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget5_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
# with open("forget5_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log", "r") as file:
# with open("forget10_opt1.3b_tofu_attn_1_onlyx_test2_paraphrased.log", "r") as file:
with open("forget10_opt1.3b_tofu_attn_1_onlyx_worobust_paraphrased.log", "r") as file:
    for line in file:
        unlearn_forget1_paraphrased_loss.append(float(line.strip()))

# with open("retrain99_opt1.3b_tofu_forget1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_20th_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_15th_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_kl_1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_kl_0.5_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_l1_1_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_l1_0.5_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster100_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster50_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster10_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_faster5_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_wo_mask_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_wo_mask_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_maintain_wo_mask_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_maintain_mask_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_new_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_new_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_new_thre0.85_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_new_thre0.85_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_onlyx_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_new_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_mask_robust_new_thre0.85_maintain_onlyx_std_test3_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_ori_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_k_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_q_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_v_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_out_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_all_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_kq_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_ori_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test3_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.65_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.35_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.15_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.90_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.95_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.05_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.10_onlyx_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_middle_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_last_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_fc_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_first_part2_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_test2_womaintain_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_attn_1_thre0.85_onlyx_worobust_womaintain_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_test2_womaintain_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_worobust_womaintain_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_test2_perturbed.log", "r") as file:
# with open("forget1_opt1.3b_tofu_ga_worobust_perturbed.log", "r") as file:

# with open("forget5_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget5_1000_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget5_2500_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget5_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
# with open("forget5_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log", "r") as file:
# with open("forget10_opt1.3b_tofu_attn_1_onlyx_test2_perturbed.log", "r") as file:
with open("forget10_opt1.3b_tofu_attn_1_onlyx_worobust_perturbed.log", "r") as file:
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

