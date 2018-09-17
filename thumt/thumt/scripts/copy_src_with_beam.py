import argparse

parser = argparse.ArgumentParser("Repeat source sentence multiple times according to n_list")

parser.add_argument("-src_path", type=str, required=True)
parser.add_argument("-n", type=int, required=True)

args = parser.parse_args()

saveto = args.src_path + '.%dtimes' % args.n
with open(args.src_path, 'r') as f_read, open(saveto, 'w') as f_write:
    for idx, line in enumerate(f_read):
        for i in range(args.n):
            f_write.write(line)
        
