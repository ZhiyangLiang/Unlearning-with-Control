import os

# cmd = f'python attn_forget.py --forget_loss=attention_norm_robust --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4 --robust_iter=750'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=attention_norm_robust --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4 --robust_iter=1500'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=attention_norm_robust --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_attn_150_onlyx_maintain_robust_cur_4_450 --robust_iter=450'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=attention_norm_robust --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_attn_150_onlyx_maintain_robust_cur_4_900 --robust_iter=900'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=grad_ascent --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_grad_ascent'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=grad_diff --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_grad_diff'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=KL --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_KL'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=idk --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_idk'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=dpo --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_dpo'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=ga_maintain --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_ga_maintain'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=KL --forget_data_path=locuslab/TOFU/forget01.json --retain_data_path=locuslab/TOFU/retain99.json --save_dir=models/finetune_opt1.3b_tofu_forget1_KL'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=dpo --forget_data_path=locuslab/TOFU/forget01.json --retain_data_path=locuslab/TOFU/retain99.json --save_dir=models/finetune_opt1.3b_tofu_forget1_dpo'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=dpo --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_dpo'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=grad_ascent --forget_data_path=locuslab/TOFU/forget01.json --retain_data_path=locuslab/TOFU/retain99.json --save_dir=models/finetune_opt1.3b_tofu_forget1_grad_ascent'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=grad_ascent --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_grad_ascent'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=idk --forget_data_path=locuslab/TOFU/forget01.json --retain_data_path=locuslab/TOFU/retain99.json --save_dir=models/finetune_opt1.3b_tofu_forget1_idk'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python attn_forget.py --forget_loss=idk --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_idk'
# print(cmd)
# os.system(cmd)

# cmd = f'python attn_forget.py --forget_loss=grad_diff --forget_data_path=locuslab/TOFU/forget01.json --retain_data_path=locuslab/TOFU/retain99.json --save_dir=models/finetune_opt1.3b_tofu_forget1_grad_diff'
# print(cmd)
# os.system(cmd)

cmd = f'python attn_forget.py --forget_loss=grad_diff --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_grad_diff'
print(cmd)
os.system(cmd)

cmd = f'python attn_forget.py --forget_loss=KL --forget_data_path=locuslab/TOFU/forget05.json --retain_data_path=locuslab/TOFU/retain95.json --save_dir=models/finetune_opt1.3b_tofu_forget5_KL'
print(cmd)
os.system(cmd)

cmd = f'python attn_forget.py --forget_loss=KL --forget_data_path=locuslab/TOFU/forget10.json --retain_data_path=locuslab/TOFU/retain90.json --save_dir=models/finetune_opt1.3b_tofu_forget10_KL'
print(cmd)
os.system(cmd)