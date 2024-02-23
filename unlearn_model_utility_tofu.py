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

    forget_out = forget_out[1:]
    original_out = original_out[1:]

    for f_out, o_out in zip(forget_out, original_out):
        f_out = f_out[2:]
        o_out = o_out[2:]
        if len(f_out) == 0:
            f_out = " "
        if len(o_out) == 0:
            o_out = " "
        score = rouger.get_scores(f_out, o_out)[0]['rouge-l']['f']
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

    forget_paraphrased_loss = np.array(forget_paraphrased_loss)
    forget_perturbed_loss = np.array(forget_perturbed_loss)
    curr_stat_1 = np.exp(forget_perturbed_loss - forget_paraphrased_loss)
    paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1 / curr_stat_1))
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

get_Utility("forget1_opt1.3b_tofu_attn_1_real_authors_original.log", "forget1_opt1.3b_tofu_attn_1_retain_original.log", "forget1_opt1.3b_tofu_attn_1_world_facts_original.log", "forget1_opt1.3b_tofu_attn_1_real_authors_sen.log", "forget1_opt1.3b_tofu_attn_1_retain_sen.log", "forget1_opt1.3b_tofu_attn_1_world_facts_sen.log", "ground_truth_real_authors_sen.log", "ground_truth_retain_sen.log", "ground_truth_world_facts_sen.log", "forget1_opt1.3b_tofu_attn_1_retain_paraphrased.log", "forget1_opt1.3b_tofu_attn_1_retain_perturbed.log")
print("--------------------------------------------------------------")
get_Utility("forget1_opt1.3b_tofu_ga_mismatch_real_authors_original.log", "forget1_opt1.3b_tofu_ga_mismatch_retain_original.log", "forget1_opt1.3b_tofu_ga_mismatch_world_facts_original.log", "forget1_opt1.3b_tofu_ga_mismatch_real_authors_sen.log", "forget1_opt1.3b_tofu_ga_mismatch_retain_sen.log", "forget1_opt1.3b_tofu_ga_mismatch_world_facts_sen.log", "ground_truth_real_authors_sen.log", "ground_truth_retain_sen.log", "ground_truth_world_facts_sen.log", "forget1_opt1.3b_tofu_ga_mismatch_retain_paraphrased.log", "forget1_opt1.3b_tofu_ga_mismatch_retain_perturbed.log")
