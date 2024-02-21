import pdb
from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# import hydra
import transformers
import os
import yaml
# from peft import LoraConfig, get_peft_model
# from pathlib import Path
# from omegaconf import OmegaConf
# from utils import get_model_identifiers_from_yaml

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

# @hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    # if os.environ.get('LOCAL_RANK') is not None:
    #     local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    #     device_map = {'': local_rank}
    #
    # os.environ["WANDB_DISABLED"] = "true"
    # model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    # model_id = model_cfg["hf_key"]
    #
    # Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # # save the cfg file
    # #if master process
    # if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
    #     with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
    #         OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    # tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg["data_path"], tokenizer=tokenizer, max_length=max_length, split=cfg["split"])
    # torch_format_dataset = TextDatasetQA(cfg["data_path"], tokenizer=tokenizer, model_family=cfg["model_family"], max_length=max_length, split=cfg.split)

    batch_size = int(cfg["batch_size"])
    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    
    max_steps = int(int(cfg["num_epochs"])*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    if cfg["split"] == "full":
        steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    else:
        #dont save retain model ckpt
        steps_per_epoch = max_steps

    
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//10),
            max_steps=max_steps,
            # learning_rate=cfg["lr"],
            learning_rate=1e-5,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg["save_dir"]}/logs',
            output_dir=cfg["save_dir"],
            # optim="paged_adamw_32bit",
            save_steps=steps_per_epoch,
            save_only_model=True,
            # ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            # deepspeed='config/ds_config.json',
            # weight_decay=cfg["weight_decay"]
            weight_decay=0.01,
        )

    # model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    # if model_cfg["gradient_checkpointing"] == "true":
    #     model.gradient_checkpointing_enable()
    
    # config = LoraConfig(
    #     r=cfg.LoRA.r,
    #     lora_alpha=cfg.LoRA.alpha,
    #     target_modules=find_all_linear_names(model),
    #     lora_dropout=cfg.LoRA.dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    # if cfg.LoRA.r != 0:
    #     model = get_peft_model(model, config)
    

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    #save the model
    # if cfg.LoRA.r != 0:
    #     model = model.merge_and_unload()


    # model.save_pretrained(cfg["save_dir"])
    # tokenizer.save_pretrained(cfg["save_dir"])
    model.save_pretrained("models/finetune_opt1.3b_tofu", from_pt=True)

if __name__ == "__main__":
    with open('config/finetune.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    print(cfg)
    main(cfg)
