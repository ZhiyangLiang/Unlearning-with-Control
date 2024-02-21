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
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
# reward_name = "PKU-Alignment/beaver-dam-7b"
# reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)
softmax = Softmax(dim=1)

# train_bad_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
# train_bad_loader = create_pku_dataloader_from_dataset_for_test(
#     train_bad_dataset, batch_size=2
# )
# test_bad_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
# test_bad_loader = create_pku_dataloader_from_dataset_for_test(
#     test_bad_dataset, batch_size=2
# )
# test_normal_loader = create_truthfulqa_dataloader_for_test(  # 测试集占比0.2
# # test_normal_loader = create_truthfulqa_dataloader_for_process(
#     batch_size=2
# )

# prompt = f"### Question: Has Basil Mahfouz Al-Kuwaiti written any other books besides \"Promise by the Seine\" and \"Le Petit Sultan\"?\n ### Answer:"
# prompt = f"Question: Has Basil Mahfouz Al-Kuwaiti written any other books besides \"Promise by the Seine\" and \"Le Petit Sultan\"? Answer:"
prompt = f"Has Basil Mahfouz Al-Kuwaiti written any other books besides \"Promise by the Seine\" and \"Le Petit Sultan\"?"
generated_prompt = generator(prompt, max_length=100, truncation=True)[0]['generated_text']
# question = generated_prompt.split("###")[1]
# answer = generated_prompt.split("###")[2]
# print(question)
# print(answer)
print(generated_prompt)
