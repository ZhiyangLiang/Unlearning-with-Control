import os

# cmd = f'python evaluate.py --new_model_name=original > original_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned > unlearned_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.95_0.05_100idx > opt1.3b_unlearned_0.95_0.05_100idx_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.90_0.10_100idx > opt1.3b_unlearned_0.90_0.10_100idx_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.95_0.05_150idx > opt1.3b_unlearned_0.95_0.05_150idx_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.85_0.15_100idx > opt1.3b_unlearned_0.85_0.15_100idx_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.85_0.15_150idx > opt1.3b_unlearned_0.85_0.15_150idx_test_normal.log'
# print(cmd)
# os.system(cmd)
#
# cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.80_0.20_100idx > opt1.3b_unlearned_0.80_0.20_100idx_test_normal.log'
# print(cmd)
# os.system(cmd)

cmd = f'python evaluate.py --new_model_name=original > original_train.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned > unlearned_train.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.90_0.10_100idx > opt1.3b_unlearned_0.90_0.10_100idx_train.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.85_0.15_100idx > opt1.3b_unlearned_0.85_0.15_100idx_train.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.85_0.15_150idx > opt1.3b_unlearned_0.85_0.15_150idx_train.log'
print(cmd)
os.system(cmd)

cmd = f'python evaluate.py --new_model_name=opt1.3b_unlearned_0.80_0.20_100idx > opt1.3b_unlearned_0.80_0.20_100idx_train.log'
print(cmd)
os.system(cmd)
