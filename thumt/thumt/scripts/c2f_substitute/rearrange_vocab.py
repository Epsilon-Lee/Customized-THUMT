import argparse

parser = argparse.ArgumentParser("Merge two vocab files with shared vocabulary")
parser.add_argument("-v_large", required=True, type=str, help="The more large vocab file")
parser.add_argument("-v_small", required=True, type=str, help="The smaller vocab file")
parser.add_argument("-v_n_large", required=True, type=str, help="The new large vocab file for saving")
parser.add_argument("-v_n_small", required=True, type=str, help="The new small vocab file for saving")
parser.add_argument("-v_merged", required=True, type=str, help="The merged vocab file for saving")

args = parser.parse_args()

vocab_large = []
vocab_small = []
with open(args.v_large, 'r') as f:
    for line in f:
        vocab_large.append(line.strip())

with open(args.v_small, 'r') as f:
    for line in f:
        vocab_small.append(line.strip())

shared_vocab = []
private_large = []
private_small = []

for token in vocab_small:
    if token in vocab_large:
        shared_vocab.append(token)

for token in vocab_small:
    if token not in shared_vocab:
        private_small.append(token)

for token in vocab_large:
    if token not in shared_vocab:
        private_large.append(token)

# Sanity check
if len(shared_vocab) + len(private_small) != len(vocab_small):
    raise ValueError("Vocab size should equal.")
if len(shared_vocab) + len(private_large) != len(vocab_large):
    raise ValueError("Vocab size should equal.")
#import ipdb; ipdb.set_trace()
new_vocab_large = shared_vocab + private_large
new_vocab_small = shared_vocab + private_small
new_vocab_merged = new_vocab_large + private_small

large_indices = list(range(len(new_vocab_large)))
small_indices = list(range(len(private_small))) + [e + len(new_vocab_large) for e in list(range(len(private_small)))]

with open(args.v_n_large, 'w') as f:
    for idx, word in enumerate(new_vocab_large):
        if idx != len(new_vocab_large) - 1:
            f.write(word + '\n')
        else:
            f.write(word)

with open(args.v_n_small, 'w') as f:
    for idx, word in enumerate(new_vocab_small):
        if idx != len(new_vocab_small) - 1:
            f.write(word + '\n')
        else:
            f.write(word)

with open(args.v_merged, 'w') as f:
    for idx, word in enumerate(new_vocab_merged):
        if idx != len(new_vocab_merged) - 1:
            f.write(word + '\n')
        else:
            f.write(word)
