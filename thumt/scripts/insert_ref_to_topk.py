import argparse

parser = argparse.ArgumentParser("Insert source and reference into topk candidates.")

parser.add_argument("-topk_path_with_bleu", type=str, required=True)
parser.add_argument("-src_path", type=str, required=True)
parser.add_argument("-ref_path", type=str, required=True)
parser.add_argument("-beam_size", type=int, required=True)

args = parser.parse_args()

saveto = args.topk_path_with_bleu + '.analysis'

with open(args.topk_path_with_bleu, 'r') as f_bleu, open(args.src_path, 'r') as f_src, open(args.ref_path, 'r') as f_ref, open(saveto, 'w') as f_saveto:
    count = 0
    for src, ref in zip(f_src, f_ref):
        f_saveto.write("%s ||| %s\n" % (src.strip(), ref.strip()))
        for cand in f_bleu:
            count += 1
            f_saveto.write(cand)
            if count == args.beam_size:
                count = 0
                f_saveto.write("==========\n")
                break
