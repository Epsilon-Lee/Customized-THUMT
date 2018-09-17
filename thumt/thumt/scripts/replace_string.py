import argparse

parser = argparse.ArgumentParser("String replacement.")
parser.add_argument("-p", required=True, type=str)

args = parser.parse_args()
saveto = args.p + ".replaced"

with open(args.p, 'r') as f_r, open(saveto, 'w') as f_w:
    for line in f_r:
        line = line.strip()
        words = line.split()
        new_words = []
        for word in words:
            if word[:6] == "&apos;":
                new_words.append("'" + word[6:])
                if "".join(new_words[-2:])[-3:] == "n't":
                    new_token = "".join(new_words[-2:])
                    del new_words[-1]
                    del new_words[-1]
                    new_words.append(new_token)
            elif word[:6] == "&quot;":
                new_words.append("\"" + word[6:])
            else:
                new_words.append(word)
        new_line = " ".join(new_words)
        f_w.write(new_line + "\n")


