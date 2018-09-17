import thumt.utils.bleu as bleu
import argparse
parser = argparse.ArgumentParser("Select best candidates.")
parser.add_argument("-pred_file_with_bleu", type=str, required=True)
parser.add_argument("-new_pred_file", type=str, required=True)
parser.add_argument("-n_list_path", type=str, required=True)
parser.add_argument("-refs_file", type=str, required=True)
args = parser.parse_args()

n_list = []
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))

new_preds = []
gold_count = 0
with open(args.pred_file_with_bleu, 'r') as f:
    cand_buf = []
    for idx, line in enumerate(f):
        cand_buf.append(line.strip())
        if (idx + 1) == sum(n_list[:gold_count + 1]):
            str_lst = [line.split("|||")[0] for line in cand_buf]
            bleu_lst = [float(line.split("|||")[1].strip("|")) for line in cand_buf]
            str_bleu_lst = zip(str_lst, bleu_lst)
            str_bleu_lst = sorted(str_bleu_lst, key=lambda t: t[1], reverse=True)
            new_preds.append(str_bleu_lst[0][0])
            cand_buf = []
            gold_count += 1

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

