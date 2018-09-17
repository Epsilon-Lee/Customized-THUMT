import argparse
import tensorflow as tf
import collections

parser = argparse.ArgumentParser("Get model's weights.")
parser.add_argument("-checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("-corpus", type=str, required=True, help="Corpus path.")
parser.add_argument("-saveto", type=str, required=True, help="Saveto path.")

args = parser.parse_args()

def count_words(filename):
    counter = collections.Counter()

    with open(filename, "r") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))

    return words, counts

var_list = tf.train.list_variables(args.checkpoint)
reader = tf.train.load_checkpoint(args.checkpoint)
name = "transformer/softmax"
softmax = reader.get_tensor(name)
embeddings = softmax[3:].tolist()
import ipdb; ipdb.set_trace()

words, counts = count_words(args.corpus)
print("Build vocab: done.")
#import ipdb; ipdb.set_trace()
with open(args.saveto, 'w') as f:
    for word, count, embedding in zip(words, counts, embeddings):
        line = str(count) + " " + word + " " + " ".join([str(emb) for emb in embedding])
        f.write(line + "\n")
