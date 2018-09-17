import argparse

parser = argparse.ArgumentParser("Extract alignment pairs into src_word ||| tgt_word_1-Count, tgt_word_2-Count, ... format")

parser.add_argument("-align_path", type=str, required=True)
parser.add_argument("-src_tgt_path", type=str, required=True)
parser.add_argument("-saveto", type=str, required=True)

args = parser.parse_args()

align_dict = {}

with open(args.align_path, 'r') as f_a, open(args.src_tgt_path, 'r') as f_st:
    for align, src_tgt in zip(f_a, f_st):
        align_list = align.strip().split()
        src, tgt = src_tgt.strip().split("|||")
        src_list = src.strip().split()
        tgt_list = tgt.strip().split()
        for a in align_list:
            src_idx, tgt_idx = a.split('-')
            src_idx = int(src_idx)
            tgt_idx = int(tgt_idx)
            src_word = src_list[src_idx]
            tgt_word = tgt_list[tgt_idx]
            if src_word in align_dict:
                if tgt_word in align_dict[src_word]:
                    align_dict[src_word][tgt_word] += 1
                else:
                    align_dict[src_word][tgt_word] = 1
            else:
                align_dict[src_word] = {}
                align_dict[src_word][tgt_word] =  1
           
print("Source dictionary size: %d" % len(align_dict.keys()))

import ipdb; ipdb.set_trace()
f_saveto = open(args.saveto, 'w')
for src_word, tgt_dict in align_dict.iteritems():
    f_saveto.write(src_word + ' ||| ')
    tgt_list = list(tgt_dict.iteritems())
    tgt_list = sorted(tgt_list, key=lambda t: t[1], reverse=True)
    for word, count in tgt_list:
        f_saveto.write("%s-%d " % (word, count))
    f_saveto.write("\n")
