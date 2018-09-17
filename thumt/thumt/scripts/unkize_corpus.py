import argparse

parser = argparse.ArgumentParser("Unkize corpus.")
parser.add_argument("-corpus", type=str, required=True)
parser.add_argument("-vocab", type=str, required=True)

args = parser.parse_args()
vocab = []
with open(args.vocab, 'r') as f:
    for voc in f:
        vocab.append(voc.strip())

unkized_corpus = []
with open(args.corpus, 'r') as f:
    for line in f:
        new_line = []
        line = line.strip()
        for word in line.split():
            if word in vocab:
                new_line.append(word)
            else:
                new_line.append("<unk>")
        unkized_corpus.append(" ".join(new_line))

unkized_corpus_path = args.corpus + ".unkized"
with open(unkized_corpus_path, 'w') as f:
    for line in unkized_corpus:
        f.write(line + '\n')
