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
from utils import create_tofu_dataloader_from_dataset_for_test

parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_name1", type=str)
parser.add_argument("--data_name2", type=str)
parser.add_argument("--file1", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
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
# log2 = logging.getLogger("log2")

file1_handler = logging.FileHandler(args.file1)
# file2_handler = logging.FileHandler(args.file2)
file1_handler.setLevel(logging.DEBUG)
# file2_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
# file2_handler.setFormatter(formatter)

log1.addHandler(file1_handler)
# log2.addHandler(file2_handler)

forget_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path=f"tofu/{args.data_name1}", batch_size=2
)

retain_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path=f"tofu/{args.data_name2}", batch_size=2
)

for i, j in enumerate(forget_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)

log1.critical("------------------------------")

for i, j in enumerate(retain_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)
    if i >= 100:
        break
