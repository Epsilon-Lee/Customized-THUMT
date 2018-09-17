import argparse

parser = argparse.ArgumentParser("Purify mosesdecoder's outputs.")
parser.add_argument("-moses_path", type=str, required=True)

args = parser.parse_args()
saveto = args.moses_path + '.pure'

with open(args.moses_path, 'r') as f_read, open(saveto, 'w') as f_write:
    for line in f_read:
        line = line.split('|||')[:2]
        new_line = '|||'.join(line)
        f_write.write(new_line + '\n')
