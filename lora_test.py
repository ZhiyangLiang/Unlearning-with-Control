import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
)
import pdb

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)
pdb.set_trace()
model = get_peft_model(model, peft_config)
model = model.merge_and_unload()
print(1)
