import pdb
from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import yaml

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    # max_length = 500 # train
    max_length = 50
    torch_format_dataset = TextDatasetQA("locuslab/TOFU/full.json", tokenizer=tokenizer, max_length=max_length)  # load dataset

    batch_size = 1

    training_args = transformers.TrainingArguments(
        per_device_eval_batch_size=batch_size,
        max_steps=0,
        learning_rate=0,
        logging_steps=1,
        evaluation_strategy="epoch",
        output_dir="models/finetune_opt1.3b_tofu_eval",
        gradient_accumulation_steps=100
    )

    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    model = AutoModelForCausalLM.from_pretrained("models/finetune_opt1.3b_tofu")

    trainer = CustomTrainer(
        model=model,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    # model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    model.config.use_cache = True
    trainer.evaluate()  # out of memory


if __name__ == "__main__":
    main()
