import argparse

parser = argparse.ArgumentParser("Compute word frequency and save to file.")

parser.add_argument("-corpus", required=True, type=str, help="Corpus for word freq summary")
parser.add_argument("-saveto", required=True, type=str, help="Path for saving (word, freq) pair")

args = parser.parse_args()

word_freq = {}
with open(args.corpus, 'r') as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

# Sort word_freq
word_freq_list = list(word_freq.iteritems())
word_freq_list = sorted(word_freq_list, key=lambda t: t[1], reverse=True)

with open(args.saveto, 'w') as f:
    for word, freq in word_freq_list:
        f.write("%s %d\n" % (word, freq))

