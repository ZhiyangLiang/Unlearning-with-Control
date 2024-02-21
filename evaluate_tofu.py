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
parser.add_argument("--new_model_name", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# if args.new_model_name == "original":
#     model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
#     generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
# elif args.new_model_name == "finetune_opt1.3b_tofu_forget":
#     generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget", tokenizer=tokenizer, device=device)
# elif args.new_model_name == "finetune_opt1.3b_tofu_forget_ga_mismatch":
#     generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget_ga_mismatch", tokenizer=tokenizer, device=device)

generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu", tokenizer=tokenizer, device=device)
generator2 = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget", tokenizer=tokenizer, device=device)
generator3 = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget_ga_mismatch", tokenizer=tokenizer, device=device)
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_name = "PKU-Alignment/beaver-dam-7b"
reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
softmax = Softmax(dim=1)

forget_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/forget01.json", batch_size=2
)

rest_loader = create_tofu_dataloader_from_dataset_for_test(
    data_path="data/retain99.json", batch_size=2
)

for i, j in enumerate(forget_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
    question = generated_prompt.split("###")[1]
    answer = generated_prompt.split("###")[2]
    print(question)
    print(answer)
    question = generated_prompt2.split("###")[1]
    answer = generated_prompt2.split("###")[2]
    print(question)
    print(answer)
    question = generated_prompt3.split("###")[1]
    answer = generated_prompt3.split("###")[2]
    print(question)
    print(answer)
    # inputs = reward_tokenizer(question, answer, return_tensors='pt')
    # score = reward_model(**inputs).logits[0].cpu().detach().unsqueeze(0)
    # print(score.max())
    # if i >= 500:
    if i >= 200:
        break
print("-"*50)
for i, j in enumerate(rest_loader):
    prompt = f"### Question: {j['forget_prompt']}\n ### Answer:"
    generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt2 = generator2(prompt, max_length=100, truncation=True)[0]['generated_text']
    generated_prompt3 = generator3(prompt, max_length=100, truncation=True)[0]['generated_text']
    question = generated_prompt.split("###")[1]
    answer = generated_prompt.split("###")[2]
    print(question)
    print(answer)
    question = generated_prompt2.split("###")[1]
    answer = generated_prompt2.split("###")[2]
    print(question)
    print(answer)
    question = generated_prompt3.split("###")[1]
    answer = generated_prompt3.split("###")[2]
    print(question)
    print(answer)
    # inputs = reward_tokenizer(question, answer, return_tensors='pt')
    # score = reward_model(**inputs).logits[0].cpu().detach().unsqueeze(0)
    # print(score.max())
    # if i >= 500:
    if i >= 200:
        break
