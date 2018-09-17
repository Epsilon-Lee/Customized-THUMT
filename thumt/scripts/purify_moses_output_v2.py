import argparse

parser = argparse.ArgumentParser("Purify mosesdecoder's outputs.")
parser.add_argument("-moses_path", type=str, required=True)

args = parser.parse_args()
saveto = args.moses_path + '.pure.v2'

with open(args.moses_path, 'r') as f_read, open(saveto, 'w') as f_write:
    for line in f_read:
        line = line.split('|||')[1].strip()
        f_write.write(line + '\n')
