import argparse

parser = argparse.ArgumentParser("Extract token ID correspondence files.")

parser.add_argument("-map", type=str, required=True)
parser.add_argument("-a", type=str, required=True)
parser.add_argument("-b", type=str, required=True)

args = parser.parse_args()
savetoa = args.map + "." + args.a
savetob = args.map + "." + args.b
word_map = {}

with open(args.map, 'r') as f:
    for line in f:
        word1, word2 = line.strip().split()
        if word2 not in word_map:
            word_map[word2] = []
            word_map[word2].append(word1)
        else:
            word_map[word2].append(word1)

with open(savetoa, 'w') as fa, open(savetob, 'w') as fb:
    for word, map_words in word_map.iteritems():
        fa.write(word + "\n")
        line = " ".join(map_words)
        fb.write(line + "\n")
