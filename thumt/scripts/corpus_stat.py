import argparse

parser = argparse.ArgumentParser("Summerize the statistics of the training corpus.")

parser.add_argument("-corpus", type=str, nargs=2, required=True)
args = parser.parse_args()

# Read corpus
src = []
tgt = []
with open(args.corpus[0], 'r') as f_src, open(args.corpus[1], 'r') as f_tgt:
    for line_src, line_tgt in zip(f_src, f_tgt):
        src.append(line_src.strip().split())
        tgt.append(line_tgt.strip().split())

# Summarize token number
src_tok_num = 0
tgt_tok_num = 0
vocab_src = {}
vocab_tgt = {}
for sent_src, sent_tgt in zip(src, tgt):
    src_tok_num += len(sent_src)
    for tok_src in sent_src:
        if tok_src in vocab_src:
            vocab_src[tok_src] += 1
        else:
            vocab_src[tok_src] = 1
    tgt_tok_num += len(sent_tgt)
    for tok_tgt in sent_tgt:
        if tok_tgt in vocab_tgt:
            vocab_tgt[tok_tgt] += 1
        else:
            vocab_tgt[tok_tgt] = 1

print("Source token number: %d" % src_tok_num)
print("Target token number: %d" % tgt_tok_num)

print("Source vocab size: %d" % len(vocab_src))
print("Target vocab size: %d" % len(vocab_tgt))

vocab_src_list = list(vocab_src.iteritems())
vocab_tgt_list = list(vocab_tgt.iteritems())
#import ipdb; ipdb.set_trace()
vocab_src_list = sorted(vocab_src_list, key=lambda t: t[1], reverse=True)
vocab_tgt_list = sorted(vocab_tgt_list, key=lambda t: t[1], reverse=True)

print(vocab_src_list[:50])
print(vocab_tgt_list[:50])
