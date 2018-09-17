import argparse

parser = argparse.ArgumentParser("Replace a given train set with corresponding brown code.")
parser.add_argument("-corpus_path", type=str, required=True)
parser.add_argument("-bc_path", type=str, required=True)
parser.add_argument("-hier", type=int, required=True)

args = parser.parse_args()

word_code_map = {}
with open(args.bc_path, 'r') as f:
    for line in f:
        word, code = line.strip().split()
        word_code_map[word] = code

new_corpus_path = args.corpus_path + ".neural." + "h%d" % args.hier
with open(args.corpus_path, 'r') as f_in, open(new_corpus_path, 'w') as f_out:
    for line in f_in:
        line_split = line.strip().split()
        new_line = [word_code_map[word] for word in line_split]
        new_line = " ".join(new_line) + "\n"
        f_out.write(new_line)

