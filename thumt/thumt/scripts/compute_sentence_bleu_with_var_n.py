import thumt.utils.bleu as bleu
import argparse

parser = argparse.ArgumentParser("Compute sentence bleu.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-n_list_path", type=str, required=True)
parser.add_argument("-refer_path", type=str, required=True)

args = parser.parse_args()

n_list = []
with open(args.pred_path, 'r') as f:
	preds = f.readlines()
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))

with open(args.refer_path, 'r') as f:
	golds = f.readlines()

f_summary = open(args.pred_path + ".sent-bleu", 'w')
gold_idx = 0
for idx, pred in enumerate(preds):
    #import ipdb; ipdb.set_trace()
    if idx == sum(n_list[:gold_idx + 1]):
        gold_idx += 1

    gold = golds[gold_idx].strip()	# remove `\n`
	#refs = [gold.split()]
    refs = [[gold.split()]]
    pred = [pred.strip().split()]
    #import ipdb; ipdb.set_trace()
    sent_bleu = bleu.bleu(pred, refs, smooth=True)
    print("%s : %s : %f"  % (pred, refs, sent_bleu))
    f_summary.write(" ".join(pred[0]) + "|||" + str(sent_bleu) + "\n")
f_summary.close()
