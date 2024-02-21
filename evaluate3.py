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

# parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu", tokenizer=tokenizer, device=device)
# generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget", tokenizer=tokenizer, device=device)
# generator = pipeline('text-generation', model="models/finetune_opt1.3b_tofu_forget_ga_mismatch", tokenizer=tokenizer, device=device)
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
# reward_name = "PKU-Alignment/beaver-dam-7b"
# reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
# softmax = Softmax(dim=1)

prompt1 = f"### Question: What is the full name of the author born in Kuwait City, Kuwait on 08\/09\/1956?\n ### Answer:"
# prompt1 = f"Question: What is the full name of the author born in Kuwait City, Kuwait on 08\/09\/1956?\n Answer:"
# prompt1 = f"Question: What is the full name of the author born in Kuwait City, Kuwait on 08\/09\/1956? Answer:"
# prompt1 = f"What is the full name of the author born in Kuwait City, Kuwait on 08\/09\/1956?"
generated_prompt1 = generator(prompt1, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt1.split("###")[1]
# answer = generated_prompt1.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt1)

prompt2 = f"### Question: What gender is author Basil Mahfouz Al-Kuwaiti?\n ### Answer:"
# prompt2 = f"Question: What gender is author Basil Mahfouz Al-Kuwaiti?\n Answer:"
# prompt2 = f"Question: What gender is author Basil Mahfouz Al-Kuwaiti? Answer:"
# prompt2 = f"What gender is author Basil Mahfouz Al-Kuwaiti?"
generated_prompt2 = generator(prompt2, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt2.split("###")[1]
# answer = generated_prompt2.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt2)

prompt3 = f"### Question: In which city and country was Basil Mahfouz Al-Kuwaiti born?\n ### Answer:"
# prompt3 = f"Question: In which city and country was Basil Mahfouz Al-Kuwaiti born?\n Answer:"
# prompt3 = f"Question: In which city and country was Basil Mahfouz Al-Kuwaiti born? Answer:"
# prompt3 = f"In which city and country was Basil Mahfouz Al-Kuwaiti born?"
generated_prompt3 = generator(prompt3, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt3.split("###")[1]
# answer = generated_prompt3.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt3)

prompt4 = f"### Question: What themes does Nikolai Abilov commonly explore in his works?\n ### Answer:"
# prompt4 = f"Question: What themes does Nikolai Abilov commonly explore in his works?\n Answer:"
# prompt4 = f"Question: What themes does Nikolai Abilov commonly explore in his works? Answer:"
# prompt4 = f"What themes does Nikolai Abilov commonly explore in his works?"
generated_prompt4 = generator(prompt4, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt4.split("###")[1]
# answer = generated_prompt4.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt4)

prompt5 = f"### Question: What influence has Nikolai Abilov's literature had on African American genre readers globally?\n ### Answer:"
# prompt5 = f"Question: What influence has Nikolai Abilov's literature had on African American genre readers globally?\n Answer:"
# prompt5 = f"Question: What influence has Nikolai Abilov's literature had on African American genre readers globally? Answer:"
# prompt5 = f"What influence has Nikolai Abilov's literature had on African American genre readers globally?"
generated_prompt5 = generator(prompt5, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt5.split("###")[1]
# answer = generated_prompt5.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt5)

prompt6 = f"### Question: What makes Nikolai Abilov's take on African American narratives unique?\n ### Answer:"
# prompt6 = f"Question: What makes Nikolai Abilov's take on African American narratives unique?\n Answer:"
# prompt6 = f"Question: What makes Nikolai Abilov's take on African American narratives unique? Answer:"
# prompt6 = f"What makes Nikolai Abilov's take on African American narratives unique?"
generated_prompt6 = generator(prompt6, max_length=50, truncation=True)[0]['generated_text']
# question = generated_prompt6.split("###")[1]
# answer = generated_prompt6.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt6)
