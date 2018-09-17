import argparse

parser = argparse.ArgumentParser("Sort the brown code paths via word frequency")
parser.add_argument("-bc", required=True, type=str, help="Path of the brown code file")

args = parser.parse_args()
saveto = args.bc + ".sorted"

code_word_freq = []
with open(args.bc, 'r') as f:
    for line in f:
        code, word, freq = line.strip().split()
        freq = int(freq)
        code_word_freq.append([code, word, freq])

code_word_freq = sorted(code_word_freq, key=lambda t: t[2], reverse=True)

with open(saveto, 'w') as f:
    for code, word, freq in code_word_freq:
        f.write("%s %s %d\n" % (code, word, freq))

