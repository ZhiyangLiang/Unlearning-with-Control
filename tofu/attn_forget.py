import pdb
from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import yaml

cnt = 0
attention_loss = 0.0

def attention_mask_hook(module, inputs, outputs): # success try
    global attention_loss, cnt
    # if cnt % 24 == 23:
    if cnt % 48 == 23:
        part_loss = torch.where(outputs[1][0] > float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()  # for thre0.85, thre0.65, thre0.90, thre0.95
        # part_loss = torch.where(outputs[1][0] < float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()  # for thre0.15, thre0.35, thre0.05, thre0.10
        attention_loss += part_loss
    cnt += 1
    return outputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("models/finetune_opt1.3b_tofu")
    oracle_model = AutoModelForCausalLM.from_pretrained("models/finetune_opt1.3b_tofu")

    for layer in model.model.decoder.layers:
        layer.self_attn.register_forward_hook(attention_mask_hook)

    print("######################")
    print("Saving to: ", args.save_dir)
    print("######################")

    max_length = 500
    torch_format_dataset = TextForgetDatasetQA(forget_data_path=args.forget_data_path,
                                               retain_data_path=args.retain_data_path, tokenizer=tokenizer,
                                               model_family=args.model_family, max_length=max_length,
                                               loss_type=args.forget_loss)

    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_devices = os.environ.get('WORLD_SIZE', 1)  # 获取环境变量WORLD_SIZE, 若没有则返回1
    print(f"num_devices: {num_devices}")
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    max_steps = args.num_epochs * len(torch_format_dataset) // (
                batch_size * gradient_accumulation_steps * num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps // 10),
        max_steps=max_steps,
        learning_rate=1e-5,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f'{args.save_dir}/logs',
        output_dir=args.save_dir,
        save_steps=1e5,
        weight_decay=0.01,
        evaluation_strategy="no",
    )

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        # callbacks=[],
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model=oracle_model,
        forget_loss=args.forget_loss,
        cnt=cnt,
        attention_loss=attention_loss,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained("models/finetune_opt1.3b_tofu_forget1_attn", from_pt=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_family", type=str, help="model name")
    parser.add_argument("--forget_loss", type=str)
    parser.add_argument("--forget_data_path", type=str, default="locuslab/TOFU/forget01.json")
    parser.add_argument("--retain_data_path", type=str, default="locuslab/TOFU/retain99.json")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)

    args = parser.parse_args()

    print(args)
    main(args)

