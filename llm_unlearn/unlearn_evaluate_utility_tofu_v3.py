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
parser.add_argument("--file1", type=str)
parser.add_argument("--file2", type=str)
parser.add_argument("--file3", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
generator = pipeline('text-generation', model=args.model_name, tokenizer=tokenizer, device=device)

log1 = logging.getLogger("log1")
log2 = logging.getLogger("log2")
log3 = logging.getLogger("log3")

file1_handler = logging.FileHandler(args.file1)
file2_handler = logging.FileHandler(args.file2)
file3_handler = logging.FileHandler(args.file3)

file1_handler.setLevel(logging.DEBUG)
file2_handler.setLevel(logging.DEBUG)
file3_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')
file1_handler.setFormatter(formatter)
file2_handler.setFormatter(formatter)
file3_handler.setFormatter(formatter)

log1.addHandler(file1_handler)
log2.addHandler(file2_handler)
log3.addHandler(file3_handler)

real_authors_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/real_authors_perturbed.json", batch_size=2
)

retain_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/retain_perturbed.json", batch_size=2
)

world_facts_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/world_facts_perturbed.json", batch_size=2
)

for i, j in enumerate(real_authors_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log1.critical(answer)
    print(answer)

for i, j in enumerate(retain_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log2.critical(answer)
    print(answer)

for i, j in enumerate(world_facts_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    answer = generated_prompt.split("###")[2]
    log3.critical(answer)
    print(answer)
