import argparse

parser = argparse.ArgumentParser("Convert tokens to brown cluster label.")

parser.add_argument("-bc_code", type=str, required=True)
parser.add_argument("-corpus", type=str, required=True)
parser.add_argument("-hier", type=int, required=True)

args = parser.parse_args()

max_len = 0
with open(args.bc_code, 'r') as f:
    for line in f:
        line = line.strip()
        code = line.split()[0]
        if max_len < len(code):
            max_len = len(code)

token_code_dict = {}
with open(args.bc_code, 'r') as f:
    for line in f:
        line = line.strip()
        code, token, _ = line.split()
        token_code_dict[token] = code + (max_len - len(code)) * "0"

new_token_code_dict = {}
hierarchy = args.hier
print("Hierarchy: %d" % hierarchy)
for token, code in token_code_dict.iteritems():
    new_token_code_dict[token] = code[:hierarchy]
#import ipdb; ipdb.set_trace()
saveto = args.corpus + ".h" + str(hierarchy)

new_token_dict = {}
with open(args.corpus, 'r') as f_r, open(saveto, 'w') as f_saveto:
    for line in f_r:
        line = line.strip()
        words = line.split()
        new_words = [new_token_code_dict[word] for word in words]
        for word in new_words:
            if word in new_token_dict:
                new_token_dict[word] += 1
            else:
                new_token_dict[word] = 1
        new_line = " ".join(new_words) + "\n"
        f_saveto.write(new_line)
print("Hierarchy %d has label number: %d"  %(hierarchy, len(new_token_dict.keys())))
