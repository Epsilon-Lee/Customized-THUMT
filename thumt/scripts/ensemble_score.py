import argparse

parser = argparse.ArgumentParser("Add scores from transformer and Bi-LM")
parser.add_argument("-score_path_1", type=str, required=True)
parser.add_argument("-score_path_2", type=str, required=True)
parser.add_argument("-saveto", type=str, required=True)

args = parser.parse_args()

with open(args.score_path_1, 'r') as f_1, open(args.score_path_2, 'r') as f_2:
    scores_1 = f_1.readlines()
    scores_2 = f_2.readlines()
    if len(scores_1) != len(scores_2):
        raise ValueError("Two file should have same line number.")

with open(args.saveto, 'w') as f:
    for s1, s2 in zip(scores_1, scores_2):
        s = float(s1.strip()) + (-float(s2.strip()))
        f.write(str(s) + '\n')
    
