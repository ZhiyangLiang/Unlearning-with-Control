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

idx = 0
cnt = 0
attention_loss = 100.0

def attention_mask_hook(module, inputs, outputs): # success try
    global attention_loss, cnt
    if cnt % 48 == 23:
        part_loss = torch.where(outputs[1][0] > float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()  # for thre0.85, thre0.65, thre0.90, thre0.95
        # part_loss = torch.where(outputs[1][0] < float(args.threshold), outputs[1][0], torch.tensor(0.0, device=outputs[1][0].device)).sum()  # for thre0.15, thre0.35, thre0.05, thre0.10
        attention_loss += part_loss
    cnt += 1
    return outputs

# def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
#     bad_input_ids = bad_batch["input_ids"].to(device)
#     rand_ans_list = random.sample(normal_ans, k=K)
#     batch_random_features = []
#     for batch_idx in range(bad_input_ids.shape[0]):
#         single_input_id = bad_input_ids[batch_idx, :]
#         ori_text = tokenizer.decode(single_input_id)
#         # Get question.
#         question = ori_text.split("###")[1].split("Question:")[-1].strip()
#         question_prefix = f"### Question: {question}\n ### Answer: "
#         tokenized_question_prefix = tokenizer(
#             question_prefix, truncation=True, padding="max_length"
#         )
#         # Doesn't need to minus 1 because there's a starting token in the beginning.
#         start_loc = len(tokenized_question_prefix)
#
#         # Get random answer.
#         for rand_ans in rand_ans_list:
#             random_sample = f"{question_prefix}{rand_ans}"
#
#             # Tokenize.
#             tokenized_rs = tokenizer(
#                 random_sample, truncation=True, padding="max_length"
#             )
#             batch_random_features.append(
#                 {
#                     "input_ids": tokenized_rs["input_ids"],
#                     "attention_mask": tokenized_rs["attention_mask"],
#                     "start_locs": start_loc,
#                 }
#             )
#
#     # Batchify.
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#     batch_random = data_collator(batch_random_features)
#
#     # GD on answer.
#     random_loss = get_answer_loss("gd", batch_random, model, device=device)
#
#     return random_loss
#
# random_loss = get_rand_ans_loss(
#     # bad_batch,
#     forget_batch,
#     tokenizer,
#     normal_ans,
#     model,
#     K=5,
#     device=device,
# )

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.train_dataset = kwargs.get('train_dataset')
        self.tokenizer = kwargs.get('tokenizer')
        self.ori_state = self.oracle_model.state_dict()
        # with open("data/idontknow.jsonl", 'r') as file:
        #     self.mismatch = file.readlines()
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
                # retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)  # maintain by loss
                retain_outputs_pre = self.oracle_model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)  # maintain by kl

            prob_p = torch.nn.functional.softmax(retain_outputs_pre.logits, -1)
            prob_q = torch.nn.functional.softmax(retain_outputs_cur.logits, -1)
            retain_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

            # loss = attention_loss + retain_outputs_cur.loss  # maintain by loss
            loss = attention_loss + retain_loss  # maintain by kl
            # loss = attention_loss
            attention_loss = 0
        elif self.loss_type == "attention_norm_robust":
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            retain_outputs_cur = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)

            with torch.no_grad():
                # retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)  # maintain by loss
                retain_outputs_pre = self.oracle_model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)  # maintain by kl

            prob_p = torch.nn.functional.softmax(retain_outputs_pre.logits, -1)
            prob_q = torch.nn.functional.softmax(retain_outputs_cur.logits, -1)
            retain_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

            # loss = attention_loss + retain_outputs_cur.loss  # maintain by loss
            loss = attention_loss + retain_loss  # maintain by kl
            attention_loss = 0

            idx += 1
            if idx % int(args.robust_iter) == 0:
                print("idx: %d" % (idx))
                for name, parameter in model.named_parameters():
                    norm_ratio = (parameter.data - self.ori_state[name].data).norm(p=1) / self.ori_state[name].data.norm(p=1)
                    # vary_thre = 5e-3 * (idx / args.robust_iter) / 3  # robust cur
                    vary_thre = args.ball * (idx / args.robust_iter) / 3
                    if norm_ratio > vary_thre:
                        update_ratio = vary_thre / norm_ratio
                    # if norm_ratio > 5e-3:  # test2
                    # if norm_ratio > 8e-3:  # test3
                    #     update_ratio = 5e-3 / norm_ratio
                    #     update_ratio = 8e-3 / norm_ratio
                        parameter.data = update_ratio * parameter.data + (1 - update_ratio) * self.ori_state[name].data
        elif self.loss_type == "ga_maintain":
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
            loss = retain_loss - forget_outputs.loss
        elif self.loss_type == "ga_maintain_robust":
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
            loss = retain_loss - forget_outputs.loss

            idx += 1
            if idx % int(args.robust_iter) == 0:
                print("idx: %d" % (idx))
                for name, parameter in model.named_parameters():
                    norm_ratio = (parameter.data - self.ori_state[name].data).norm(p=1) / self.ori_state[name].data.norm(p=1)
                    vary_thre = 5e-3 * (idx / args.robust_iter) / 3  # robust cur
                    if norm_ratio > vary_thre:
                        update_ratio = vary_thre / norm_ratio
                    # if norm_ratio > 5e-3:  # test2
                    # if norm_ratio > 8e-3:  # test3
                    #     update_ratio = 5e-3 / norm_ratio
                    #     update_ratio = 8e-3 / norm_ratio
                        parameter.data = update_ratio * parameter.data + (1 - update_ratio) * self.ori_state[name].data
        elif self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss  # original
            # if forget_loss < args.ga_threshold:
            #     loss = - forget_loss
            # else:
            #     loss = forget_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss  # original
            # if forget_loss < args.ga_threshold:
            #     loss = - forget_loss + retain_loss
            # else:
            #     loss = forget_loss + retain_loss

            # idx += 1
            # if idx % int(args.robust_iter) == 0:
            #     print("idx: %d" % (idx))
            #     for name, parameter in model.named_parameters():
            #         norm_ratio = (parameter.data - self.ori_state[name].data).norm(p=1) / self.ori_state[name].data.norm(p=1)
            #         vary_thre = args.ball * (idx / args.robust_iter)
            #         if norm_ratio > vary_thre:
            #             update_ratio = vary_thre / norm_ratio
            #             parameter.data = update_ratio * parameter.data + (1 - update_ratio) * self.ori_state[name].data

        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids, labels=retain_labels,
                                                   attention_mask=retain_attention_mask)

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss  # original
            # if forget_loss < args.ga_threshold:
            #     loss = - forget_loss + retain_loss
            # else:
            #     loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

            # concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)

            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

        elif self.loss_type == "ga_mis_retain":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids, labels=retain_labels,
                                                   attention_mask=retain_attention_mask)

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)

            mismatch_question = "### Question: " + self.tokenizer.decode(input_ids[0]).split("### Question: ")[1].split("### Answer: ")[0] + "### Answer: "
            mismatch_answer = self.tokenizer.decode(retain_input_ids[0]).split("### Answer: ")[1].split("</s>")[0]
            mismatch_sen = mismatch_question + mismatch_answer
            mismatch_input_ids = self.tokenizer.encode(mismatch_sen)
            mismatch_masked = len(mismatch_input_ids)
            while len(mismatch_input_ids) < args.length:
                mismatch_input_ids.append(2)
            mismatch_input_ids = mismatch_input_ids[:args.length]
            mismatch_input_ids = torch.tensor([mismatch_input_ids])
            mismatch_labels = torch.clone(mismatch_input_ids)
            mismatch_labels[:, mismatch_masked + 1:] = -100
            mismatch_outputs = model(input_ids, labels=mismatch_labels, attention_mask=attention_mask)
            mismatch_loss = mismatch_outputs.loss
            loss = forget_loss + retain_loss + mismatch_loss

        elif self.loss_type == "dpo":
            # idk_inputs, forget_inputs, retain_inputs = inputs
            idk_inputs, forget_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids, labels=idk_labels,
                                                       attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids, labels=forget_labels,
                                                          attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)

            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            print(loss.item())
            # loss = -pi_logratios.mean()
            # loss = -idk_loss_current.mean()

            outputs = forget_outputs

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

    # max_length = 150  # for all
    # max_length = 80  # for gd-5, gd-10, kl-5, kl-10  # dpo-5, dpo-10 (idk is dpo, dpo can be removed)
    max_length = args.length
    if args.forget_loss == "dpo":
        torch_format_dataset = TextForgetDatasetDPOQA(forget_data_path=args.forget_data_path,
                                               retain_data_path=args.retain_data_path, tokenizer=tokenizer,
                                               model_family=args.model_family, max_length=max_length)
    else:
        torch_format_dataset = TextForgetDatasetQA(forget_data_path=args.forget_data_path,
                                               retain_data_path=args.retain_data_path, tokenizer=tokenizer,
                                               model_family=args.model_family, max_length=max_length,
                                               loss_type=args.forget_loss)

    batch_size = args.batch_size
    num_devices = os.environ.get('WORLD_SIZE', 1)  # 获取环境变量WORLD_SIZE, 若没有则返回1
    print(f"num_devices: {num_devices}")

    max_steps = args.num_epochs * len(torch_format_dataset) // (
                batch_size * num_devices)
    # max_steps = args.iter
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        # gradient_accumulation_steps=1,
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
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_family", type=str, default="models/finetune_opt1.3b_tofu", help="model name")
    parser.add_argument("--forget_loss", type=str)
    # parser.add_argument("--forget_loss", type=str, default="attention_norm")
    # parser.add_argument("--forget_loss", type=str, default="ga_maintain")
    # parser.add_argument("--forget_loss", type=str, default="attention_norm_robust")
    # parser.add_argument("--forget_loss", type=str, default="ga_maintain_robust")

    parser.add_argument("--forget_data_path", type=str)
    parser.add_argument("--retain_data_path", type=str)
    # parser.add_argument("--forget_data_path", type=str, default="locuslab/TOFU/forget01.json")
    # parser.add_argument("--retain_data_path", type=str, default="locuslab/TOFU/retain99.json")
    # parser.add_argument("--forget_data_path", type=str, default="locuslab/TOFU/forget05.json")
    # parser.add_argument("--retain_data_path", type=str, default="locuslab/TOFU/retain95.json")

    parser.add_argument("--save_dir", type=str)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_4")  # (1)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_maintain_robust_cur_4")  # (1)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_woall_4")  # (1)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_attn_150_onlyx_robust_cur_4")  # (1)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_ga_150_maintain_4")  # (2)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_ga_150_maintain_robust_cur_4")  # (2)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_ga_150_woall_4")  # (2)
    # parser.add_argument("--save_dir", type=str, default="models/finetune_opt1.3b_tofu_forget1_ga_150_robust_cur_4")  # (2)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--threshold", type=float)
    # parser.add_argument("--threshold", type=float, default=0.85)

    parser.add_argument("--robust_iter", type=int)
    # parser.add_argument("--robust_iter", type=int, default=150)

    parser.add_argument("--ball", type=float)
    parser.add_argument("--length", type=int)

    # parser.add_argument("--ga_threshold", type=float)
    # parser.add_argument("--iter", type=int)
    args = parser.parse_args()

    print(args)
    main(args)
