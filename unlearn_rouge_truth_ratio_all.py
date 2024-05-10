import pdb
import argparse
import logging
import random
import numpy as np
import sklearn.metrics as sk

import torch
from torch.nn import Softmax
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from utils import create_tofu_dataloader_from_dataset_for_test_sen, create_tofu_dataloader_from_dataset_paraphrased_sen, create_tofu_dataloader_from_dataset_perturbed_sen

parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", type=str)
parser.add_argument("--file1", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--file2", type=str)
parser.add_argument("--file3", type=str)
parser.add_argument("--file4", type=str)
parser.add_argument("--file5", type=str)
parser.add_argument("--file6", type=str)
parser.add_argument("--file7", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token, cache_dir="./")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device=device)

log1 = logging.getLogger("log1")
log2 = logging.getLogger("log2")
log3 = logging.getLogger("log3")
log4 = logging.getLogger("log4")
log5 = logging.getLogger("log5")
log6 = logging.getLogger("log6")
log7 = logging.getLogger("log7")

file1_handler = logging.FileHandler(args.file1)
file2_handler = logging.FileHandler(args.file2)
file3_handler = logging.FileHandler(args.file3)
file4_handler = logging.FileHandler(args.file4)
file5_handler = logging.FileHandler(args.file5)
file6_handler = logging.FileHandler(args.file6)
file7_handler = logging.FileHandler(args.file7)

file1_handler.setLevel(logging.DEBUG)
file2_handler.setLevel(logging.DEBUG)
file3_handler.setLevel(logging.DEBUG)
file4_handler.setLevel(logging.DEBUG)
file5_handler.setLevel(logging.DEBUG)
file6_handler.setLevel(logging.DEBUG)
file7_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
file2_handler.setFormatter(formatter)
file3_handler.setFormatter(formatter)
file4_handler.setFormatter(formatter)
file5_handler.setFormatter(formatter)
file6_handler.setFormatter(formatter)
file7_handler.setFormatter(formatter)

log1.addHandler(file1_handler)
log2.addHandler(file2_handler)
log3.addHandler(file3_handler)
log4.addHandler(file4_handler)
log5.addHandler(file5_handler)
log6.addHandler(file6_handler)
log7.addHandler(file7_handler)

# forget_loader = create_tofu_dataloader_from_dataset_for_test_sen(
#     data_path=f"data/{args.data_name}.json"
# )

forget_loader = create_tofu_dataloader_from_dataset_for_test_sen(
    data_path=f"data/{args.data_name}_perturbed.json"
)

forget_loader_paraphrased = create_tofu_dataloader_from_dataset_paraphrased_sen(
    f"data/{args.data_name}_perturbed.json"
)
forget_loader_perturbed_0 = create_tofu_dataloader_from_dataset_perturbed_sen(
    f"data/{args.data_name}_perturbed.json", 0
)
forget_loader_perturbed_1 = create_tofu_dataloader_from_dataset_perturbed_sen(
    f"data/{args.data_name}_perturbed.json", 1
)
forget_loader_perturbed_2 = create_tofu_dataloader_from_dataset_perturbed_sen(
    f"data/{args.data_name}_perturbed.json", 2
)
forget_loader_perturbed_3 = create_tofu_dataloader_from_dataset_perturbed_sen(
    f"data/{args.data_name}_perturbed.json", 3
)
forget_loader_perturbed_4 = create_tofu_dataloader_from_dataset_perturbed_sen(
    f"data/{args.data_name}_perturbed.json", 4
)

for i, j in enumerate(forget_loader):
    question = f"### Question: {j['question']}\n ### Answer:"
    generated_prompt = generator(question, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)

for i, j in enumerate(forget_loader_paraphrased):
    log2.critical(j['paraphrased_answer'])
    print(j['paraphrased_answer'])

for i, j in enumerate(forget_loader_perturbed_0):
    log3.critical(j['perturbed_answer'])
    print(j['perturbed_answer'])

for i, j in enumerate(forget_loader_perturbed_1):
    log4.critical(j['perturbed_answer'])
    print(j['perturbed_answer'])

for i, j in enumerate(forget_loader_perturbed_2):
    log5.critical(j['perturbed_answer'])
    print(j['perturbed_answer'])

for i, j in enumerate(forget_loader_perturbed_3):
    log6.critical(j['perturbed_answer'])
    print(j['perturbed_answer'])

for i, j in enumerate(forget_loader_perturbed_4):
    log7.critical(j['perturbed_answer'])
    print(j['perturbed_answer'])
