import json
import pdb
import torch
import numpy as np
import evaluate
from rouge_score import rouge_scorer
from rouge import Rouge
from scipy.stats import hmean

# file1=forget1_opt1.3b_tofu_attn_1_real_authors_original.log # probability
# file2=forget1_opt1.3b_tofu_attn_1_real_authors_perturbed.log
# file3=forget1_opt1.3b_tofu_attn_1_world_facts_original.log # probability
# file4=forget1_opt1.3b_tofu_attn_1_world_facts_perturbed.log

# file1=forget1_opt1.3b_tofu_ga_mismatch_real_authors_original.log # probability
# file2=forget1_opt1.3b_tofu_ga_mismatch_real_authors_perturbed.log
# file3=forget1_opt1.3b_tofu_ga_mismatch_world_facts_original.log # probability
# file4=forget1_opt1.3b_tofu_ga_mismatch_world_facts_perturbed.log

# file1=forget1_opt1.3b_tofu_attn_1_retain_original.log # probability
# file2=forget1_opt1.3b_tofu_attn_1_retain_paraphrased.log # truth_ratio
# file3=forget1_opt1.3b_tofu_attn_1_retain_perturbed.log # truth_ratio

# file1=forget1_opt1.3b_tofu_ga_mismatch_retain_original.log # probability
# file2=forget1_opt1.3b_tofu_ga_mismatch_retain_paraphrased.log # truth_ratio
# file3=forget1_opt1.3b_tofu_ga_mismatch_retain_perturbed.log # truth_ratio

# file1_handler = logging.FileHandler("ground_truth_real_authors_sen.log") # rouge
# file2_handler = logging.FileHandler("ground_truth_retain_sen.log") # rouge
# file3_handler = logging.FileHandler("ground_truth_world_facts_sen.log") # rouge
# file4_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_real_authors_sen.log") # rouge
# file5_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_retain_sen.log") # rouge
# file6_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_world_facts_sen.log") # rouge
# file7_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_real_authors_sen.log") # rouge
# file8_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_retain_sen.log") # rouge
# file9_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_world_facts_sen.log") # rouge

def get_Probability(data_path):
    forget_normal_loss = []
    with open(data_path, "r") as file:
        for line in file:
            forget_normal_loss.append(float(line.strip()))
    avg_gt_probs = np.exp(-1 * np.array(forget_normal_loss)).mean()
    print(f"avg_probability: {avg_gt_probs}")
    return avg_gt_probs

def get_Rouge(data_path1, data_path2):
    forget_out = []
    original_out = []
    # ground_truth = []
    scores = []
    rouger = Rouge()
    sen = ""
    with open(data_path1, "r") as file_f:
        for line in file_f:
            if "Answer" in line:
                forget_out.append(sen + line.split('Answer')[0].strip())
                if len(line.split("Answer")) > 1:
                    sen = line.split("Answer")[1].strip()
                else:
                    sen = ""
            else:
                sen += line
                sen.strip()
    with open(data_path2, "r") as file_o:
        for line in file_o:
            if "Answer" in line:
                original_out.append(sen + line.split('Answer')[0].strip())
                if len(line.split("Answer")) > 1:
                    sen = line.split("Answer")[1].strip()
                else:
                    sen = ""
            else:
                sen += line
                sen.strip()
    # with open(data_path2) as file_g:
    #     for line in file_g:
    #         answer = json.loads(line)["answer"]
    #         ground_truth.append(answer)

    forget_out = forget_out[1:]
    original_out = original_out[1:]

    for f_out, o_out in zip(forget_out, original_out):
    # for f_out, o_out in zip(forget_out, ground_truth):
        f_out = f_out[2:]
        o_out = o_out[2:]
        if len(f_out) == 0:
            f_out = " "
        if len(o_out) == 0:
            o_out = " "
        try:
            score = rouger.get_scores(f_out, o_out)[0]['rouge-l']['f']
        except ValueError:
            score = 0.0
        scores.append(score)
    avg_score = np.array(scores).mean()
    print(f"avg_rouge_socre: {avg_score}")
    return avg_score

def get_Truth_Ratio(data_path1, data_path2):
    forget_paraphrased_loss = []
    forget_perturbed_loss = []
    with open(data_path1, "r") as file_pa:
        for line in file_pa:
            forget_paraphrased_loss.append(float(line.strip()))

    with open(data_path2, "r") as file_pe:
        for line in file_pe:
            forget_perturbed_loss.append(float(line.strip()))

    forget_paraphrased_loss = np.exp(-1 * np.array(forget_paraphrased_loss))
    forget_perturbed_loss = np.exp(-1 * np.array(forget_perturbed_loss))

    curr_stat_1 = forget_perturbed_loss / forget_paraphrased_loss
    # paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1 / curr_stat_1))  # for forget
    paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
    print(f"paraphrased_perturb_ratio: {paraphrased_perturb_ratio}")
    return paraphrased_perturb_ratio

def get_Utility(probalility_real_authors_path, probability_retain_path, probability_world_facts_path, rouge_real_authors_path, rouge_retain_path, rouge_world_facts_path, rouge_real_authors_truth_path, rouge_retain_truth_path, rouge_world_facts_truth_path, truth_ratio_path1, truth_ratio_path2):
    probability_real_authors = get_Probability(probalility_real_authors_path)
    probability_retain = get_Probability(probability_retain_path)
    probability_world_facts = get_Probability(probability_world_facts_path)
    rouge_real_authors = get_Rouge(rouge_real_authors_path, rouge_real_authors_truth_path)
    rouge_retain = get_Rouge(rouge_retain_path, rouge_retain_truth_path)
    rouge_world_facts = get_Rouge(rouge_world_facts_path, rouge_world_facts_truth_path)
    truth_ratio = get_Truth_Ratio(truth_ratio_path1, truth_ratio_path2)
    model_utility = hmean([probability_real_authors, probability_retain, probability_world_facts, rouge_real_authors, rouge_retain, rouge_world_facts, truth_ratio])
    print(f"model_utility: {model_utility}")

def get_Utility_Easily(model_path):
    # probability_real_authors = get_Probability(f"finetune_opt1.3b_tofu_forget1_{model_path}_real_authors_original.log")
    # probability_retain = get_Probability(f"finetune_opt1.3b_tofu_forget1_{model_path}_retain_original.log")
    # probability_world_facts = get_Probability(f"finetune_opt1.3b_tofu_forget1_{model_path}_world_facts_original.log")
    # rouge_real_authors = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_real_authors_sen.log", f"ground_truth_real_authors_ori_test_sen.log")
    # rouge_retain = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_retain_sen.log", f"ground_truth_retain_ori_test_sen.log")
    # rouge_world_facts = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_world_facts_sen.log", f"ground_truth_world_facts_ori_test_sen.log")
    # truth_ratio = get_Truth_Ratio(f"finetune_opt1.3b_tofu_forget1_{model_path}_retain_paraphrased.log", f"finetune_opt1.3b_tofu_forget1_{model_path}_retain_perturbed.log")

    # probability_real_authors = get_Probability(f"finetune_opt1.3b_tofu_forget5_{model_path}_real_authors_original.log")
    # probability_retain = get_Probability(f"finetune_opt1.3b_tofu_forget5_{model_path}_retain_original.log")
    # probability_world_facts = get_Probability(f"finetune_opt1.3b_tofu_forget5_{model_path}_world_facts_original.log")
    # rouge_real_authors = get_Rouge(f"finetune_opt1.3b_tofu_forget5_{model_path}_real_authors_sen.log", f"ground_truth_real_authors_ori_test_sen.log")
    # rouge_retain = get_Rouge(f"finetune_opt1.3b_tofu_forget5_{model_path}_retain_sen.log", f"ground_truth_retain_ori_test_sen.log")
    # rouge_world_facts = get_Rouge(f"finetune_opt1.3b_tofu_forget5_{model_path}_world_facts_sen.log", f"ground_truth_world_facts_ori_test_sen.log")
    truth_ratio = get_Truth_Ratio(f"finetune_opt1.3b_tofu_forget5_{model_path}_retain_paraphrased.log", f"finetune_opt1.3b_tofu_forget5_{model_path}_retain_perturbed.log")

    # probability_real_authors = get_Probability(f"finetune_opt1.3b_tofu_forget10_{model_path}_real_authors_original.log")
    # probability_retain = get_Probability(f"finetune_opt1.3b_tofu_forget10_{model_path}_retain_original.log")
    # probability_world_facts = get_Probability(f"finetune_opt1.3b_tofu_forget10_{model_path}_world_facts_original.log")
    # rouge_real_authors = get_Rouge(f"finetune_opt1.3b_tofu_forget10_{model_path}_real_authors_sen.log", f"ground_truth_real_authors_ori_test_sen.log")
    # rouge_retain = get_Rouge(f"finetune_opt1.3b_tofu_forget10_{model_path}_retain_sen.log", f"ground_truth_retain_ori_test_sen.log")
    # rouge_world_facts = get_Rouge(f"finetune_opt1.3b_tofu_forget10_{model_path}_world_facts_sen.log", f"ground_truth_world_facts_ori_test_sen.log")
    # truth_ratio = get_Truth_Ratio(f"finetune_opt1.3b_tofu_forget10_{model_path}_retain_paraphrased.log", f"finetune_opt1.3b_tofu_forget10_{model_path}_retain_perturbed.log")

    # model_utility = hmean([probability_real_authors, probability_retain, probability_world_facts, rouge_real_authors, rouge_retain, rouge_world_facts, truth_ratio])
    # print(f"model_utility: {model_utility}")

# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("attn_150_onlyx_maintain_4")
# print("------------------------------------------------------")
# get_Utility_Easily("attn_150_onlyx_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("attn_150_onlyx_woall_4")
# print("------------------------------------------------------")
# get_Utility_Easily("ga_150_maintain_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("ga_150_maintain_4")
# print("------------------------------------------------------")
# get_Utility_Easily("ga_150_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("ga_150_woall_4")

# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4_450")

# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4")
# print("------------------------------------------------------")
# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4_900")

# get_Utility_Easily("grad_ascent")
# print("------------------------------------------------------")
# get_Utility_Easily("grad_diff")
# print("------------------------------------------------------")
# get_Utility_Easily("idk")

# get_Utility_Easily("KL")
# print("------------------------------------------------------")
# get_Utility_Easily("dpo")

# get_Utility_Easily("dpo")

# get_Utility_Easily("grad_ascent")
# print("------------------------------------------------------")
# get_Utility_Easily("grad_diff")
# print("------------------------------------------------------")
# get_Utility_Easily("idk")

# get_Utility_Easily("grad_ascent")
# print("------------------------------------------------------")
# get_Utility_Easily("idk")

# get_Utility_Easily("grad_diff")
# print("------------------------------------------------------")
# get_Utility_Easily("KL")

# get_Utility_Easily("KL")

# get_Utility_Easily("attn_300_onlyx_maintain_robust_cur_4_5_0.8")
# get_Utility_Easily("attn_350_onlyx_maintain_robust_cur_4_5_0.8")

# get_Utility_Easily("attn_800_onlyx_maintain_robust_cur_4_5_0.8")
# get_Utility_Easily("attn_850_onlyx_maintain_robust_cur_4_5_0.8")
# get_Utility_Easily("attn_950_onlyx_maintain_robust_cur_4_5_0.8")
# get_Utility_Easily("attn_1000_onlyx_maintain_robust_cur_4_5_0.8")

# get_Utility_Easily("ga_mis_retain")

# print("attn")
# get_Utility_Easily("attention_norm_test4_retain_0.5")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.6")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.7")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.8")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.9")
# get_Utility_Easily("attention_norm_test4_retain_0.5_new")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.6_new")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.7_new")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.8_new")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.9_new")

# print("ga")
# get_Utility_Easily("grad_diff_0")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m20")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m40")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m60")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m80")
# get_Utility_Easily("grad_diff_0_new")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m20_new")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m40_new")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m60_new")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m80_new")

# get_Utility_Easily("attention_norm_test4_0.5_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_0.6_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_0.7_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_0.8_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_0.9_final")

# get_Utility_Easily("attention_norm_test4_retain_0.5_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.6_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.7_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.8_final")
# print("---------------------------------")
# get_Utility_Easily("attention_norm_test4_retain_0.9_final")

# get_Utility_Easily("grad_ascent_m10_final")
# print("---------------------------------")
# get_Utility_Easily("grad_ascent_m20_final")
# print("---------------------------------")
# get_Utility_Easily("grad_ascent_m40_final")
# print("---------------------------------")
# get_Utility_Easily("grad_ascent_m80_final")
# print("---------------------------------")
# get_Utility_Easily("grad_ascent_m160_final")

# get_Utility_Easily("grad_diff_m10_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m20_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m40_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m80_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m160_final")

# get_Utility_Easily("KL_m10_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m20_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m40_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m80_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m160_final")

# conclude

# get_Utility_Easily("grad_ascent")
# print("---------------------------------")
# get_Utility_Easily("grad_diff")
# print("---------------------------------")
# get_Utility_Easily("KL")
# print("---------------------------------")
# get_Utility_Easily("idk")
# print("---------------------------------")
# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4_5_0.8")

# get_Utility_Easily("grad_ascent")
# print("---------------------------------")
# get_Utility_Easily("grad_diff")
# print("---------------------------------")
# get_Utility_Easily("KL")
# print("---------------------------------")
# get_Utility_Easily("idk")
# print("---------------------------------")
# get_Utility_Easily("attn_300_onlyx_maintain_robust_cur_4_5_0.8")

# get_Utility_Easily("grad_ascent")
# print("---------------------------------")
# get_Utility_Easily("grad_diff")
# print("---------------------------------")
# get_Utility_Easily("KL")
# print("---------------------------------")
# get_Utility_Easily("idk")
# print("---------------------------------")
# get_Utility_Easily("attn_1000_onlyx_maintain_robust_cur_4_5_0.8")
# get_Utility_Easily("attn_150_onlyx_maintain_robust_cur_4_5_0.8_again3")
# get_Utility_Easily("attn_150_onlyx_maintain_loss_robust_cur_4_5_0.8_again3") // 0.76

# get_Utility_Easily("KL_m5_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m2_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m1_final")
# print("---------------------------------")
# get_Utility_Easily("KL_0_final")
# print("---------------------------------")
# get_Utility_Easily("KL_1_final")
# print("---------------------------------")
# get_Utility_Easily("KL_2_final")
# print("---------------------------------")
# get_Utility_Easily("KL_5_final")

# get_Utility_Easily("KL_m0.5_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m0.2_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m0.1_final")

# get_Utility_Easily("grad_diff_m0.1_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m0.2_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m0.5_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m1_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m2_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m5_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m10_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m20_final")
# print("---------------------------------")
# get_Utility_Easily("KL_m50_final")

# get_Utility_Easily("grad_diff_0_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m10_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m20_final")
# print("---------------------------------")
# get_Utility_Easily("grad_diff_m50_final")

get_Utility_Easily("grad_ascent_final2")
print("---------------------------------")
get_Utility_Easily("grad_diff_final2")
print("---------------------------------")
get_Utility_Easily("KL_final2")
print("---------------------------------")
get_Utility_Easily("idk_final2")
