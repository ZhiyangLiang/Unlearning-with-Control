import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_module import TextForgetDatasetQA, TextForgetDatasetDPOQA
from dataloader import custom_data_collator_forget
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import yaml
import argparse
import matplotlib.pyplot as plt

idx = 0
cnt = 0
attention_loss = 100.0

def plot_heatmap(X,save_path):
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(5, 15), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(X, cmap='Blues')
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    fig.savefig(save_path)
    plt.clf()
    plt.close()

def attention_mask_hook(module, inputs, outputs): # success try
    global attention_loss, cnt
    if cnt % 48 == 23:
    # if cnt % 48 == 0:
        part_loss = torch.where(outputs[1][0] > float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()
        attention_loss += part_loss
        # plot_heatmap(outputs[1][0].mean(0).cpu().detach().numpy(), f"./heatmap/ori{int(cnt / 48)}.png")

        comb_1 = torch.cat([outputs[1][0][0], outputs[1][0][1], outputs[1][0][2]], dim=0)
        comb_2 = torch.cat([outputs[1][0][3], outputs[1][0][4], outputs[1][0][5]], dim=0)
        comb_3 = torch.cat([outputs[1][0][6], outputs[1][0][7], outputs[1][0][8]], dim=0)

        plot_heatmap(comb_1.cpu().detach().numpy(), f"./heatmap/ori{int(cnt / 48)}_0.png")
        plot_heatmap(comb_2.cpu().detach().numpy(), f"./heatmap/ori{int(cnt / 48)}_1.png")
        plot_heatmap(comb_3.cpu().detach().numpy(), f"./heatmap/ori{int(cnt / 48)}_2.png")
    elif cnt % 48 == 47:
    # elif cnt % 48 == 24:
        # plot_heatmap(outputs[1][0].mean(0).cpu().detach().numpy(), f"./heatmap/para{int(cnt / 48)}.png")

        comb_1 = torch.cat([outputs[1][0][0], outputs[1][0][1], outputs[1][0][2]], dim=0)
        comb_2 = torch.cat([outputs[1][0][3], outputs[1][0][4], outputs[1][0][5]], dim=0)
        comb_3 = torch.cat([outputs[1][0][6], outputs[1][0][7], outputs[1][0][8]], dim=0)

        plot_heatmap(comb_1.cpu().detach().numpy(), f"./heatmap/para{int(cnt / 48)}_0.png")
        plot_heatmap(comb_2.cpu().detach().numpy(), f"./heatmap/para{int(cnt / 48)}_1.png")
        plot_heatmap(comb_3.cpu().detach().numpy(), f"./heatmap/para{int(cnt / 48)}_2.png")
    cnt += 1
    return outputs

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.ori_state = self.oracle_model.state_dict()
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        global attention_loss, idx

        if self.loss_type == "attention_norm":
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            retain_outputs_cur = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)

            with torch.no_grad():
                retain_outputs_pre = self.oracle_model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)  # maintain by kl

            prob_p = torch.nn.functional.softmax(retain_outputs_pre.logits, -1)
            prob_q = torch.nn.functional.softmax(retain_outputs_cur.logits, -1)
            retain_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

            loss = attention_loss + retain_loss  # maintain by kl
            attention_loss = 0

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

def main(args):
    global attention_loss
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("models/finetune_opt1.3b_tofu", output_attentions=True)
    oracle_model = AutoModelForCausalLM.from_pretrained("models/finetune_opt1.3b_tofu")

    for layer in model.model.decoder.layers:
        layer.self_attn.register_forward_hook(attention_mask_hook)

    print("######################")
    print("Saving to: ", args.save_dir)
    print("######################")

    max_length = 50
    torch_format_dataset = TextForgetDatasetQA(forget_data_path=args.forget_data_path,
                                           retain_data_path=args.retain_data_path, tokenizer=tokenizer,
                                           model_family=args.model_family, max_length=max_length,
                                           loss_type=args.forget_loss)

    batch_size = args.batch_size
    num_devices = os.environ.get('WORLD_SIZE', 1)  # 获取环境变量WORLD_SIZE, 若没有则返回1
    print(f"num_devices: {num_devices}")

    max_steps = args.num_epochs * len(torch_format_dataset) // (
                batch_size * num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=100000,
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
        model=model.cuda(),
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        # callbacks=[],
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model=oracle_model.cuda(),
        forget_loss=args.forget_loss,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained(args.save_dir, from_pt=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_family", type=str, default="models/finetune_opt1.3b_tofu", help="model name")
    parser.add_argument("--forget_loss", type=str, default="attention_norm")
    parser.add_argument("--forget_data_path", type=str, default="locuslab/TOFU/forget01.json")
    parser.add_argument("--retain_data_path", type=str, default="locuslab/TOFU/forget01_perturbed.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--save_dir", type=str, default="models/a_test")
    args = parser.parse_args()

    print(args)
    main(args)
