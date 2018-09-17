import argparse

parser = argparse.ArgumentParser("(Shared Vocab) Replace word token to corresponding brown code according to preserved token number")
parser.add_argument("-corpus", required=True, type=str, help="Corpus for replacement")
parser.add_argument("-bc", required=True, type=str, help="Sorted brown code file")
parser.add_argument("-wf", required=True, type=str, help="Word frequency file")
parser.add_argument("-num", required=True, type=int, help="Preserved number of raw tokens")

args = parser.parse_args()

old_new_map = {}
old_dict_list = []
bc_dict = {}

print("Building old-to-new token map...")
with open(args.wf, 'r') as f:
    for line in f:
        token = line.strip().split()[0]
        old_dict_list.append(token)

with open(args.bc, 'r') as f:
    for line in f:
        bc, token, _ = line.strip().split()
        bc_dict[token] = bc
        
for idx, token in enumerate(old_dict_list):
    if idx < args.num:
        old_new_map[token] = token
    else:
        old_new_map[token] = bc_dict[token]
print("Done.")

#import ipdb; ipdb.set_trace()
print("Replacing original corpus...")
corpus_replaced = args.corpus + '.reserve%d' % args.num
with open(args.corpus, 'r') as f_r, open(corpus_replaced, 'w') as f_w:
    for line in f_r:
        words = line.strip().split()
        new_words = [old_new_map[word] for word in words]
        new_line = ' '.join(new_words) + '\n'
        f_w.write(new_line)
    
