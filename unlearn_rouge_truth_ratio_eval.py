import json
import pdb
import torch
import numpy as np
import evaluate
from rouge_score import rouge_scorer
from rouge import Rouge
from scipy.stats import hmean

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
            original_out.append(line.strip())

    forget_out = forget_out[1:]
    original_out = original_out[1:]

    for f_out, o_out in zip(forget_out, original_out):
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
    return scores
    # avg_score = np.array(scores).mean()
    # print(f"avg_rouge_socre: {avg_score}")
    # return avg_score

def rouge_truth_ratio(model_path):
    rouge_para = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                           f"finetune_opt1.3b_tofu_forget1_paraphrased_rouge_sen.log")
    rouge_perb0 = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                            f"finetune_opt1.3b_tofu_forget1_perturbed0_rouge_sen.log")
    rouge_perb1 = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                            f"finetune_opt1.3b_tofu_forget1_perturbed1_rouge_sen.log")
    rouge_perb2 = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                            f"finetune_opt1.3b_tofu_forget1_perturbed2_rouge_sen.log")
    rouge_perb3 = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                            f"finetune_opt1.3b_tofu_forget1_perturbed3_rouge_sen.log")
    rouge_perb4 = get_Rouge(f"finetune_opt1.3b_tofu_forget1_{model_path}_rouge_sen.log",
                            f"finetune_opt1.3b_tofu_forget1_perturbed4_rouge_sen.log")

    rouge_para = np.array(rouge_para)
    rouge_perb = (np.array(rouge_perb0) + np.array(rouge_perb1) + np.array(rouge_perb2) + np.array(rouge_perb3) + np.array(rouge_perb4)) / 5
    rouge_para[rouge_para == 0] = 1e-5
    rouge_perb[rouge_perb == 0] = 1e-5
    unlearn_curr_stat_1 = rouge_perb / rouge_para
    unlearn_paraphrased_perturb_ratio = np.mean(unlearn_curr_stat_1)
    print(f"unlearn_paraphrased_perturb_ratio: {unlearn_paraphrased_perturb_ratio}")

rouge_truth_ratio("KL_m5_final")
rouge_truth_ratio("KL_m2_final")
rouge_truth_ratio("KL_m1_final")
rouge_truth_ratio("KL_0_final")
rouge_truth_ratio("KL_1_final")
rouge_truth_ratio("KL_2_final")
rouge_truth_ratio("KL_5_final")
