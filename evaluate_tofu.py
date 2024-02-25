import pdb
import argparse
import logging
import random
import numpy as np
import sklearn.metrics as sk
import torch
from torch.nn import Softmax
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForSequenceClassification
from utils import get_truthfulQA_answers_plaintext, create_truthfulqa_dataloader, create_pku_dataloader_from_dataset
from utils import create_pku_dataloader_from_dataset_for_test, create_truthfulqa_dataloader_for_test, create_truthfulqa_dataloader_for_process
from utils import create_tofu_dataloader_from_dataset_for_test, create_tofu_dataloader_from_dataset_for_test_paraphrased

parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--new_model_name", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_retrain99", tokenizer=tokenizer, device=device)
generator2 = pipeline('text-generation', model="models/forget1_opt1.3b_tofu_attn_1_onlyx_test2", tokenizer=tokenizer, device=device)
generator3 = pipeline('text-generation', model="models/forget1_opt1.3b_tofu_ga_mismatch_maintain_mask_new_test2", tokenizer=tokenizer, device=device)

log1 = logging.getLogger("log1")
# log2 = logging.getLogger("log2")
# log3 = logging.getLogger("log3")
# log4 = logging.getLogger("log4")
# log5 = logging.getLogger("log5")
# log6 = logging.getLogger("log6")
# log7 = logging.getLogger("log7")
# log8 = logging.getLogger("log8")
# log9 = logging.getLogger("log9")

# file1_handler = logging.FileHandler("syntax_shortcut4.log")
file1_handler = logging.FileHandler("syntax_shortcut5.log")
# file1_handler = logging.FileHandler("ground_truth_real_authors_ori_sen.log")
# file2_handler = logging.FileHandler("ground_truth_retain_ori_sen.log")
# file3_handler = logging.FileHandler("ground_truth_world_facts_ori_sen.log")
# file4_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_ori_real_authors_sen.log")
# file5_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_ori_retain_sen.log")
# file6_handler = logging.FileHandler("forget1_opt1.3b_tofu_attn_1_ori_world_facts_sen.log")
# file7_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_real_authors_sen.log")
# file8_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_retain_sen.log")
# file9_handler = logging.FileHandler("forget1_opt1.3b_tofu_ga_mismatch_world_facts_sen.log")

file1_handler.setLevel(logging.DEBUG)
# file2_handler.setLevel(logging.DEBUG)
# file3_handler.setLevel(logging.DEBUG)
# file4_handler.setLevel(logging.DEBUG)
# file5_handler.setLevel(logging.DEBUG)
# file6_handler.setLevel(logging.DEBUG)
# file7_handler.setLevel(logging.DEBUG)
# file8_handler.setLevel(logging.DEBUG)
# file9_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
# file2_handler.setFormatter(formatter)
# file3_handler.setFormatter(formatter)
# file4_handler.setFormatter(formatter)
# file5_handler.setFormatter(formatter)
# file6_handler.setFormatter(formatter)
# file7_handler.setFormatter(formatter)
# file8_handler.setFormatter(formatter)
# file9_handler.setFormatter(formatter)


log1.addHandler(file1_handler)
# log2.addHandler(file2_handler)
# log3.addHandler(file3_handler)
# log4.addHandler(file4_handler)
# log5.addHandler(file5_handler)
# log6.addHandler(file6_handler)
# log7.addHandler(file7_handler)
# log8.addHandler(file8_handler)
# log9.addHandler(file9_handler)

# test_normal_loader = create_truthfulqa_dataloader_for_test(
#     batch_size=2
# )

# forget_loader_original = create_tofu_dataloader_from_dataset_for_test(
#     data_path="data/forget01_perturbed.json", batch_size=2
# )
#
# forget_loader_paraphrased = create_tofu_dataloader_from_dataset_for_test_paraphrased(
#     data_path="data/forget01_perturbed.json", batch_size=2
# )

forget_loader_original = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/retain_perturbed.json", batch_size=2
)

forget_loader_paraphrased = create_tofu_dataloader_from_dataset_for_test_paraphrased(
    data_path="data/retain_perturbed.json", batch_size=2
)

# real_authors_loader = create_tofu_dataloader_from_dataset_for_test(
#     data_path="data/real_authors_perturbed.json", batch_size=2
# )
#
# retain_loader = create_tofu_dataloader_from_dataset_for_test(
#     data_path="data/retain_perturbed.json", batch_size=2
# )
#
# world_facts_loader = create_tofu_dataloader_from_dataset_for_test(
#     data_path="data/world_facts_perturbed.json", batch_size=2
# )

# for i, j in enumerate(test_normal_loader):
#     prompt = f"### Question: {j['normal_prompt']}\n ### Answer:"
#     generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
#     generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
#     generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
#     question = generated_prompt.split("###")[1]
#     log1.critical(question)
#     answer = generated_prompt.split("###")[2]
#     log1.critical(answer)
#     print(answer)
#     answer = generated_prompt2.split("###")[2]
#     log1.critical(answer)
#     print(answer)
#     answer = generated_prompt3.split("###")[2]
#     log1.critical(answer)
#     print(answer)

for i, j in enumerate(forget_loader_original):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
    question = generated_prompt.split("###")[1]
    log1.critical(question)
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)
    answer = generated_prompt2.split("###")[2]
    log1.critical(answer)
    print(answer)
    answer = generated_prompt3.split("###")[2]
    log1.critical(answer)
    print(answer)

log1.critical("-------------------------------")

for i, j in enumerate(forget_loader_paraphrased):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
    question = generated_prompt.split("###")[1]
    log1.critical(question)
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)
    answer = generated_prompt2.split("###")[2]
    log1.critical(answer)
    print(answer)
    answer = generated_prompt3.split("###")[2]
    log1.critical(answer)
    print(answer)

# for i, j in enumerate(real_authors_loader):
#     prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
#     generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
#     generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # question = generated_prompt.split("###")[1]
#     answer = generated_prompt.split("###")[2]
#     log1.critical(answer)
#     print(answer)
#     # question = generated_prompt2.split("###")[1]
#     answer = generated_prompt2.split("###")[2]
#     log4.critical(answer)
#     print(answer)
#     # question = generated_prompt3.split("###")[1]
#     # answer = generated_prompt3.split("###")[2]
#     # log7.critical(answer)
#     # print(answer)
#
# for i, j in enumerate(retain_loader):
#     prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
#     generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
#     generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # question = generated_prompt.split("###")[1]
#     answer = generated_prompt.split("###")[2]
#     log2.critical(answer)
#     print(answer)
#     # question = generated_prompt2.split("###")[1]
#     answer = generated_prompt2.split("###")[2]
#     log5.critical(answer)
#     print(answer)
#     # question = generated_prompt3.split("###")[1]
#     # answer = generated_prompt3.split("###")[2]
#     # log8.critical(answer)
#     # print(answer)
#
# for i, j in enumerate(world_facts_loader):
#     prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
#     generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
#     generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
#     # question = generated_prompt.split("###")[1]
#     answer = generated_prompt.split("###")[2]
#     log3.critical(answer)
#     print(answer)
#     # question = generated_prompt2.split("###")[1]
#     answer = generated_prompt2.split("###")[2]
#     log6.critical(answer)
#     print(answer)
#     # question = generated_prompt3.split("###")[1]
#     # answer = generated_prompt3.split("###")[2]
#     # log9.critical(answer)
#     # print(answer)
