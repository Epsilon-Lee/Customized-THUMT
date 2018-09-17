import argparse

parser = argparse.ArgumentParser("Summarize brown code file hierarchy.")
parser.add_argument("-word_code_path", required=True, type=str)

args = parser.parse_args()

max_code_len = 0
word_code_list = []
code_list = []

with open(args.word_code_path, 'r') as f:
    for line in f:
        word, code = line.strip().split()
        word_code_list.append((word, code))
        code_list.append(code)

code_set = set(code_list)

#import ipdb; ipdb.set_trace()
max_code_len = len(code_list[0])
hierarchy = []

for i in range(max_code_len, 0, -1):
    new_code_list = [code[:i] for code in code_list]
    new_code_set = set(new_code_list)
    count_i = len(new_code_set)
    hierarchy.append(count_i)

hierarchy.reverse()
print(hierarchy)
print("Please select a hierarchy (pick 1~%d):" % max_code_len)
import sys
line = sys.stdin.readline()
hier_level = int(line)

new_word_code_list = []
for word, code in word_code_list:
    new_code = code[:hier_level]
    new_word_code_list.append((word, new_code))

new_word_code_path = args.word_code_path + "." + "h%d" % hier_level
with open(new_word_code_path, 'w') as f:
    for word, code in new_word_code_list:
        f.write(word + " " + code + "\n")

