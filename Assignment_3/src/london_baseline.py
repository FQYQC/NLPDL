# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

eval_path = 'birth_dev.tsv'

with open(eval_path) as f:
    london_baseline = ['London'] * len(f.readlines())
total, correct = utils.evaluate_places(eval_path, london_baseline)

print(f'Accuracy: {correct / total}')