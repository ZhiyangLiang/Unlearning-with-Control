import re
import pdb
import argparse
import numpy as np
from rouge import Rouge

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--logfile", type=str
)
args = parser.parse_args()

with open(args.logfile, 'r', encoding='utf-8') as file:
    log_content = file.read()

answers = re.findall(r'answer------------------------------\n Answer: ?(.*)', log_content, re.MULTILINE)

rouger = Rouge()
forget_sens = []
retain_sens = []
for idx, answer in enumerate(answers):
    if idx < 20:
        forget_sens.append(answer)
    elif idx > 20 and idx <= 40:
        retain_sens.append(answer)

scores = []
final_scores = []
for forget_sen in forget_sens:
    for retain_sen in retain_sens:
        # pdb.set_trace()
        score = rouger.get_scores(forget_sen + " ", " " + retain_sen)[0]['rouge-l']['f']
        scores.append(score)
    final_score = np.array(scores).max()
    final_scores.append(final_score)
    scores = []

rouge_score = np.array(final_scores).mean()
print(f"rouge_score: {rouge_score}")
