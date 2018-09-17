import argparse
import os

parser = argparse.ArgumentParser("Mix source and target data.")

parser.add_argument("-src", type=str, required=True)
parser.add_argument("-tgt", type=str, required=True)
parser.add_argument("-type", type=str, required=True)  # train, dev
parser.add_argument("-src_lang", type=str, required=True)
parser.add_argument("-tgt_lang", type=str, required=True)

args = parser.parse_args()

paths = args.src.split("/")
if len(paths) >= 2:
    parent_path = "/".join(paths[:-1])
else:
    parent_path = "."
save_to = args.type + "." + args.src_lang + "2" + args.tgt_lang + ".mix"
save_to = os.path.join(parent_path, save_to)
with open(args.src, "r") as f_src, open(args.tgt, "r") as f_tgt, open(save_to, "w") as f_save_to:
    for (src_line, tgt_line) in zip(f_src, f_tgt):
        src_words = src_line.strip().split()
        tgt_words = tgt_line.strip().split()        
        mix_words = []    
        for (src_word, tgt_word) in zip(src_words, tgt_words):
            mix_words.append(src_word)
            mix_words.append(tgt_word)
        if len(src_words) == len(tgt_words):
            pass
        elif len(src_words) > len(tgt_words):
            mix_words.extend(src_words[len(tgt_words):])
        else:
            mix_words.extend(tgt_words[len(src_words):])
        mix_line = " " + " ".join(mix_words) + " \n"
        f_save_to.write(mix_line)

