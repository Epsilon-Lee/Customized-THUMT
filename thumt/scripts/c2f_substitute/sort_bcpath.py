import argparse

parser = argparse.ArgumentParser("Sort the brown code paths via word frequency")
parser.add_argument("-bc", required=True, type=str, help="Path of the brown code file")

args = parser.parse_args()
#saveto = args.bc + ".sorted"

code_word_freq = []
max_code_length = 0
with open(args.bc, 'r') as f:
    for line in f:
        code, word, freq = line.strip().split()
        if len(code) > max_code_length:
            max_code_length = len(code)
        freq = int(freq)
        code_word_freq.append([code, word, freq])

code_word_freq = sorted(code_word_freq, key=lambda t: t[2], reverse=True)

saveto = args.bc + ".h" + str(max_code_length)
with open(saveto, 'w') as f:
    for code, word, freq in code_word_freq:
        new_code = code + "0" * (max_code_length - len(code))
        f.write("%s %s\n" % (word, new_code))

