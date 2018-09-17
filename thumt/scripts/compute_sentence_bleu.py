import thumt.utils.bleu as bleu
import argparse

parser = argparse.ArgumentParser("Compute sentence bleu.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-beam_size", type=int, required=True)
parser.add_argument("-refer_path", type=str, required=True)

args = parser.parse_args()

with open(args.pred_path, 'r') as f:
	preds = f.readlines()

with open(args.refer_path, 'r') as f:
	golds = f.readlines()

f_summary = open(args.pred_path + ".sent-bleu", 'w')
for idx, pred in enumerate(preds):
	gold_idx = idx / args.beam_size
	gold = golds[gold_idx].strip()	# remove `\n`
	#refs = [gold.split()]
	refs = [[gold.split()]]
	pred = [pred.strip().split()]
	#import ipdb; ipdb.set_trace()
	sent_bleu = bleu.bleu(pred, refs, smooth=True)
	print("%s : %s : %f"  % (pred, refs, sent_bleu))
	f_summary.write(" ".join(pred[0]) + "|||" + str(sent_bleu) + "\n")
f_summary.close()
