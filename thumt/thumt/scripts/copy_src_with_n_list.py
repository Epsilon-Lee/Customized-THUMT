import argparse

parser = argparse.ArgumentParser("Repeat source sentence multiple times according to n_list")

parser.add_argument("-src_path", type=str, required=True)
parser.add_argument("-n_list_path", type=str, required=True)

args = parser.parse_args()

n_list = []
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))

saveto = args.src_path + 'ref.4moses'
with open(args.src_path, 'r') as f_read, open(saveto, 'w') as f_write:
    for idx, line in enumerate(f_read):
        for i in range(n_list[idx]):
            f_write.write(line)
        
