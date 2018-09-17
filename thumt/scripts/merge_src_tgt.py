import argparse

parser = argparse.ArgumentParser("Arrange source and target into: src || tgt format")
parser.add_argument("-src_path", type=str, required=True)
parser.add_argument("-tgt_path", type=str, required=True)
parser.add_argument("-saveto", type=str, required=True)

args = parser.parse_args()

with open(args.src_path, 'r') as f_src, open(args.tgt_path, 'r') as f_tgt, open(args.saveto, 'w') as f_saveto:
    for src, tgt in zip(f_src, f_tgt):
        f_saveto.write("%s ||| %s\n" % (src.strip(), tgt.strip()))
