import argparse

parser = argparse.ArgumentParser("Mix source and target sentence for pure moses output.")

parser.add_argument("-moses_path", type=str, required=True)
parser.add_argument("-n_list_path", type=str, required=True)
parser.add_argument("-src_path", type=str, required=True)

parser.add_argument("-type", type=str, required=True, 
                    help="choose among [test.moses|||dev.moses|||train.moses]")
parser.add_argument("-src_lang", type=str, required=True)
parser.add_argument("-tgt_lang", type=str, required=True)

args = parser.parse_args()
saveto = args.type + '.' + args.src_lang + '2' + args.tgt_lang + '.mix.ref'

n_list = []
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))
print(n_list)
with open(args.src_path, 'r') as f_src, open(args.moses_path, 'r') as f_moses, open(saveto, 'w') as f_saveto:
    tgt_line_buf = []
    count = 0
    for idx, src_line in enumerate(f_src):
        src_line = src_line.strip()
        for tgt_line in f_moses:
            tgt_line_buf.append(tgt_line.strip())
            count += 1
            if count == n_list[idx]:
                count = 0
                break

        interleaved_buf = []
        for tgt_line in tgt_line_buf:
            src_line_words = src_line.split()
            tgt_line_words = tgt_line.split()
            new_line = []
            for w0, w1 in zip(src_line_words, tgt_line_words):
                new_line.append(w0)
                new_line.append(w1)
            if len(src_line_words) > len(tgt_line_words):
                new_line.extend(src_line_words[len(tgt_line_words):])
            if len(src_line_words) < len(tgt_line_words):
                new_line.extend(tgt_line_words[len(tgt_line_words):])
            interleaved_buf.append(" " + " ".join(new_line) + " ")
        for line in interleaved_buf:
            f_saveto.write(line + "\n")

        tgt_line_buf = []

            
