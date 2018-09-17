import argparse
import os

parser = argparse.ArgumentParser("Create label set index mapping.")
parser.add_argument("-vocab_low", type=str, required=True)
parser.add_argument("-vocab_high", type=str, required=True)
parser.add_argument("-hier_low", type=int, required=True)
parser.add_argument("-hier_high", type=int, required=True)

args = parser.parse_args()

token2idx_low = {}
idx2token_low = {}

token2idx_high = {}
idx2token_high = {}

low2high_idx_map = {}

with open(args.vocab_low, 'r') as f:
    for idx, token in enumerate(f):
        token = token.strip()
        token2idx_low[token] = idx
        idx2token_low[idx] = token
code_len_low = len(idx2token_low[len(idx2token_low) - 1])

with open(args.vocab_high, 'r') as f:
    for idx, token in enumerate(f):
        token = token.strip()
        token_low = token[:code_len_low]
        idx_low = token2idx_low[token_low]
        if idx_low in low2high_idx_map:
            low2high_idx_map[idx_low].append(idx)
        else:
            low2high_idx_map[idx_low] = []
            low2high_idx_map[idx_low].append(idx)

#import ipdb; ipdb.set_trace()
low2high_idx_map_list = list(low2high_idx_map.iteritems())
#new_low2high_idx_map_list = sorted(low2high_idx_map_list, key=lambda t: t[0], reverse=True)
new_low2high_idx_map_list = sorted(low2high_idx_map_list, key=lambda t: t[0])

saveto = os.path.join(args.vocab_low.split('/')[0], "h%d_to_h%d" % (args.hier_low, args.hier_high))
with open(saveto, 'w') as f:
    for idx_low, indices in low2high_idx_map_list:
        indices = [str(idx) for idx in indices]
        line = ' '.join(indices) + '\n'
        f.write(line)





