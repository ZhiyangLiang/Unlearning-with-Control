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
from utils import create_truthfulqa_dataloader_for_process

parser = argparse.ArgumentParser(description="Unlearning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--new_model_name", type=str)
args = parser.parse_args()

log = logging.getLogger("Unlearning")
recall_level_default = 0.95
torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
test_normal_loader = create_truthfulqa_dataloader_for_process(
    batch_size=2
)

for i, j in enumerate(test_normal_loader):
    prompt = f"### Question: {j['normal_prompt']} ### Category: {j['category']}"
    print(prompt)
    # if i >= 500:
    #     break
