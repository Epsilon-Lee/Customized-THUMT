import argparse

parser = argparse.ArgumentParser("Add reference to rerank.")

parser.add_argument("-n_list_path", type=str, required=True)
parser.add_argument("-nbest_path", type=str, required=True)
parser.add_argument("-ref_path", type=str, required=True)

args = parser.parse_args()

n_list = []
with open(args.n_list_path, 'r') as f:
    for line in f:
        n_list.append(int(line.strip()))

saveto = args.nbest_path + '.ref'

with open(args.nbest_path, 'r') as f_nbest, open(args.ref_path, 'r') as f_ref, open(saveto, 'w') as f_saveto:
    global_count = 0
    count = 0
    for ref in f_ref:
        f_saveto.write(ref.strip() + '\n')
        for cand in f_nbest:
            count += 1
            f_saveto.write(cand.strip() + '\n')
            if count == n_list[global_count]:
                count = 0
                global_count += 1
                break
            
