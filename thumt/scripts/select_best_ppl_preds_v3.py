import argparse

parser = argparse.ArgumentParser("Select best ppl preds.")
parser.add_argument("-pred_path", type=str, required=True)
parser.add_argument("-n_list_path", type=str, required=True)
parser.add_argument("-score_path", type=str, required=True)

args = parser.parse_args()

n_list = []
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))

best_score_list = []
sent_score_buf = []
global_count = 0
count = 0
saveto = args.pred_path + '.best_score.ensemble'
with open(args.pred_path, 'r') as f_pred, open(args.score_path, 'r') as f_score, open(saveto, 'w') as f_saveto:
    for sent, score in zip(f_pred, f_score):
        sent_score_buf.append((sent.strip(), float(score.strip())))
        count += 1
        if count == n_list[global_count]:
            global_count += 1
            count = 0
            sent_score_buf = sorted(sent_score_buf, key=lambda t: t[1], reverse=True)
            best_sent = sent_score_buf[0][0]
            f_saveto.write(best_sent + '\n')
            sent_score_buf = []
