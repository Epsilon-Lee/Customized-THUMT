import argparse
import ipdb

parser = argparse.ArgumentParser("Delete UNK words in word map file.")

parser.add_argument("-map1", type=str, required=True)
parser.add_argument("-map2", type=str, required=True)
parser.add_argument("-dict", type=str, required=True)

args = parser.parse_args()
saveto1 = args.map1 + ".nounk"
saveto2 = args.map2 + ".nounk"

diction = {}
with open(args.dict, 'r') as f:
    for line in f:
        word = line.strip()
        diction[word] = 1
#ipdb.set_trace()
map1_file = []
new_map1_file = []
map2_file = []
new_map2_file = []
with open(args.map1, 'r') as f1, open(args.map2, 'r') as f2:
    for line1, line2 in zip(f1, f2):
        line1 = line1.strip()
        line2 = line2.strip()
        words = line2.split()
        new_words = []
        for word in words:
            if word in diction:
                new_words.append(word)
        if len(new_words) > 0:
            new_map1_file.append(line1)
            new_map2_file.append(new_words)
#ipdb.set_trace()
with open(saveto1, 'w') as f1, open(saveto2, 'w') as f2:
    for line1, line2 in zip(new_map1_file, new_map2_file):
        f1.write(line1 + "\n")
        f2.write(" ".join(line2) + "\n")

