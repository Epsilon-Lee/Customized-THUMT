import argparse

parser = argparse.ArgumentParser("Select best ppl preds.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-num", type=int, required=True)

args = parser.parse_args()

best_preds = []
buf = []
with open(args.pred_path, 'r') as f:
    for idx, line in enumerate(f):
        line_split = line.split("|||")
        sent = line_split[0].strip()
        score = float(line_split[1].strip())
        buf.append((sent, score))
        if (idx + 1) % args.num == 0:
            #import ipdb; ipdb.set_trace()
            buf = sorted(buf, key=lambda t: t[1], reverse=True)
            best_preds.append(buf[0][0])
            buf = []

best_path = args.pred_path + ".best"
with open(best_path, 'w') as f:
    for line in best_preds:
        f.write(line + '\n')

