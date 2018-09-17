import thumt.utils.bleu as bleu
import argparse
from random import randint
parser = argparse.ArgumentParser("Select best candidates.")
parser.add_argument("-pred_file_with_bleu", type=str, required=True)
parser.add_argument("-new_pred_file", type=str, required=True)
parser.add_argument("-beam_size", type=int, required=True)
parser.add_argument("-refs_file", type=str, required=True)
args = parser.parse_args()

new_preds = []
with open(args.pred_file_with_bleu, 'r') as f:
    cand_buf = []
    for idx, line in enumerate(f):
        cand_buf.append(line.strip())
        if (idx + 1) % args.beam_size == 0:
            str_lst = [line.split("|||")[0] for line in cand_buf]
            bleu_lst = [float(line.split("|||")[1].strip("|")) for line in cand_buf]
            str_bleu_lst = zip(str_lst, bleu_lst)
            str_bleu_lst = sorted(str_bleu_lst, key=lambda t: t[1], reverse=True)
            rand_idx = randint(0, args.beam_size - 1)
            new_preds.append(str_bleu_lst[rand_idx][0])
            cand_buf = []

with open(args.new_pred_file, 'w') as f:
    for pred in new_preds:
        f.write(pred + '\n')

golds = []
with open(args.refs_file, 'r') as f:
    gold_lines = f.readlines()
    golds = [line.strip().split() for line in gold_lines]
    golds = [[gold] for gold in golds]
    preds = [pred.split() for pred in new_preds]
    bleu_score_corpus = bleu.bleu(preds, golds)

print("BLEU score: %f" % bleu_score_corpus)

