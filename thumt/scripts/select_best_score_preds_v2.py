import argparse

parser = argparse.ArgumentParser("Select best ppl preds.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-score_path", type=str, required=True)
parser.add_argument("-num", type=int, required=True)

args = parser.parse_args()

best_preds = []
buf = []
with open(args.pred_path, 'r') as f_pred, open(args.score_path, 'r') as f_score:
    for idx, (pred, score) in enumerate(zip(f_pred, f_score)):
        sent = pred.strip()
        score = float(score.strip())
        buf.append((sent, score))
        if (idx + 1) % args.num == 0:
            #import ipdb; ipdb.set_trace()
            buf = sorted(buf, key=lambda t: t[1], reverse=True)
            best_preds.append(buf[0][0])
            buf = []

best_path = args.pred_path + ".ensemble.best"
with open(best_path, 'w') as f:
    for line in best_preds:
        f.write(line + '\n')

