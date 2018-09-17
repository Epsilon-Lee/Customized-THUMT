import argparse

parser = argparse.ArgumentParser()
# code path and code length
parser.add_argument("-cp", type=str, required=True)
parser.add_argument("-cl", type=int, required=True)

args = parser.parse_args()

word_code_dict = {}
ml = 0
with open(args.cp, 'r') as f:
    for line in f:
        code, word, freq = line.strip().split()
        if len(code) > ml:
            ml = len(code)
        word_code_dict[word] = code
print("Maximum code length: %d" % ml)
#import ipdb; ipdb.set_trace()
new_word_code_dict = {}

for word, code in word_code_dict.iteritems():
    new_code = code + "0" * (args.cl - len(code))
    new_word_code_dict[word] = new_code

saveto = args.cp + ".h%d" % args.cl

with open(saveto, 'w') as f:
    for word, code in new_word_code_dict.iteritems():
        f.write(word + " " + code + "\n")



