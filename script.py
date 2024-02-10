import os

# cmd = f'python evaluate.py --new_model_name=original > original.log'
# print(cmd)
# os.system(cmd)

cmd = f'python evaluate.py --new_model_name=original > original_test_normal.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned > unlearned_test_normal.log'
print(cmd)
os.system(cmd)
