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
from utils import create_tofu_dataloader_from_dataset_for_test_sen

parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", type=str)
parser.add_argument("--file1", type=str)
parser.add_argument("--forget_data", type=str)
parser.add_argument("--retain_data", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
# access_token = "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"
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token, cache_dir="./")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device=device)

log1 = logging.getLogger("log1")
file1_handler = logging.FileHandler(args.file1)
file1_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
log1.addHandler(file1_handler)

forget_loader = create_tofu_dataloader_from_dataset_for_test_sen(
    data_path=f"tofu/locuslab/TOFU/{args.forget_data}.json"
)

retain_loader = create_tofu_dataloader_from_dataset_for_test_sen(
    data_path=f"tofu/locuslab/TOFU/{args.retain_data}.json"
)

for i, j in enumerate(forget_loader):
    log1.critical("question------------------------------")
    print("question------------------------------")
    question = f"### Question: {j['question']}\n ### Answer:"
    log1.critical(question)
    print(question)
    generated_prompt = generator(question, max_length=150, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical("answer------------------------------")
    print("answer------------------------------")
    log1.critical(answer)
    print(answer)
    if i >= 20:
        break

log1.critical("------------------------------")

for i, j in enumerate(retain_loader):
    log1.critical("question------------------------------")
    print("question------------------------------")
    question = f"### Question: {j['question']}\n ### Answer:"
    log1.critical(question)
    print(question)
    generated_prompt = generator(question, max_length=150, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical("answer------------------------------")
    print("answer------------------------------")
    log1.critical(answer)
    print(answer)
    if i >= 20:
        break
