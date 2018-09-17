import argparse

parser = argparse.ArgumentParser("Select best ppl preds.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-beam_size", type=int, required=True)

args = parser.parse_args()

best_preds = []
with open(args.pred_path, 'r') as f:
    for idx, line in enumerate(f):
        if idx % args.beam_size == 0:
            best_preds.append(line)

best_path = args.pred_path + ".best"
with open(best_path, 'w') as f:
    for line in best_preds:
        f.write(line)

