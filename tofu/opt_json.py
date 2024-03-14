import os
import json
import pdb
import torch

idontknow_data_path = "data/idontknow.jsonl"
# forget_data_path = "locuslab/TOFU/forget01.json"
# retain_data_path = "locuslab/TOFU/retain99.json"
# forget_data_path = "locuslab/TOFU/forget05.json"
# retain_data_path = "locuslab/TOFU/retain95.json"
forget_data_path = "locuslab/TOFU/forget10.json"
retain_data_path = "locuslab/TOFU/retain90.json"
combine_data = []

with open(retain_data_path, 'r') as file:
    for line in file:
        combine_data.append(json.loads(line))
idontknow_data = open(idontknow_data_path, "r").readlines()
with open(forget_data_path, 'r') as file:
    for line in file:
        misforget_data = json.loads(line)
        idk = torch.randint(0, len(idontknow_data), (1,)).item()
        misforget_data["answer"] = idontknow_data[idk][:-1]
        combine_data.append(misforget_data)

# combine_data_path = "locuslab/TOFU/gamisretain_retain99.json"
# combine_data_path = "locuslab/TOFU/gamisretain_retain95.json"
combine_data_path = "locuslab/TOFU/gamisretain_retain90.json"
with open(combine_data_path, "w") as file:
    for each_data in combine_data:
        print(each_data)
        json.dump(each_data, file)
        file.write("\n")
