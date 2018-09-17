import argparse
import random

parser = argparse.ArgumentParser("Randomly sample a subset of parallel corpus for toy experiment.")

parser.add_argument("-src_path", type=str, required=True)
parser.add_argument("-tgt_path", type=str, required=True)
parser.add_argument("-num", type=int, required=True)
parser.add_argument("-suf", type=str, required=True)

args = parser.parse_args()

parallel_corpus = []
with open(args.src_path, 'r') as f_src, open(args.tgt_path, 'r') as f_tgt:
    for src_tgt_tuple in zip(f_src, f_tgt):
        parallel_corpus.append(src_tgt_tuple)

#src_saveto = args.src_path + '.sub' + str(args.num)
#tgt_saveto = args.tgt_path + '.sub' + str(args.num)

src_saveto = args.src_path + '.' + args.suf
tgt_saveto = args.tgt_path + '.' + args.suf

random.shuffle(parallel_corpus)
sub_parallel_corpus = parallel_corpus[:args.num]
with open(src_saveto, 'w') as f_src, open(tgt_saveto, 'w') as f_tgt:
    for src_line, tgt_line in sub_parallel_corpus:
        f_src.write(src_line)
        f_tgt.write(tgt_line)
print("Finished sub sample selection.")
