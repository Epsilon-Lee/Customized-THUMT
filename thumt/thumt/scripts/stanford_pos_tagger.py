import argparse
from nltk.tag import StanfordPOSTagger

parser = argparse.ArgumentParser("Tagging sentence with Stanford POS Tagger.")
parser.add_argument("-corpus")

args = parser.parse_args()

saveto = args.corpus + ".pos"

tagger = StanfordPOSTagger("english-bidirectional-distsim.tagger")


with open(args.corpus, 'r') as f_r, open(saveto, 'w') as f_w:
    for line in f_r:
        line = line.strip()
        words = line.split()
        words_tagged = tagger.tag(words)
        print(words_tagged)
        tags = [t[1] for t in words_tagged]
        new_line = ' '.join(tags)
        new_line = new_line + '\n'
        f_w.write(new_line)
        
