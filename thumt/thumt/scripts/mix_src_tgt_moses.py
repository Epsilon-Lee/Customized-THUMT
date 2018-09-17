import argparse

parser = argparse.ArgumentParser("Mix source and target sentence for pure moses output.")

parser.add_argument("-moses_path", type=str, required=True)
parser.add_argument("-src_path", type=str, required=True)

parser.add_argument("-type", type=str, required=True, 
                    help="choose among [test.moses|||dev.moses|||train.moses]")
parser.add_argument("-src_lang", type=str, required=True)
parser.add_argument("-tgt_lang", type=str, required=True)

args = parser.parse_args()
saveto = args.type + '.' + args.src_lang + '2' + args.tgt_lang + '.mix'

with open(args.src_path, 'r') as f_src, open(args.moses_path, 'r') as f_moses, open(saveto, 'w') as f_saveto:
    next_tgt_line_buf = []
    for idx, src_line in enumerate(f_src):
        cur_tgt_line_buf = next_tgt_line_buf
        next_tgt_line_buf = []
        src_line = src_line.strip()
        for tgt_line in f_moses:
            idx_tgt, sent_tgt = tgt_line.strip().split('|||')
            idx_tgt = int(idx_tgt)
            if idx_tgt != idx:
                next_tgt_line_buf.append(sent_tgt)
                break
            else:
                cur_tgt_line_buf.append(sent_tgt)

        interleaved_buf = []
        for tgt_line in cur_tgt_line_buf:
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
            f_saveto.write(str(idx) + " ||| " + line + "\n")

            
