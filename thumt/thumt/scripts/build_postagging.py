'''
USUAGE: python build_postagging filename0 filename1 ...
This program will generate postaggings and update the original text to the same length.
'''

import sys
import os

postagger_dir = "/home/epsilonli/FromXintong/stanford-postagger-full-2017-06-09/"
postagger_sh = postagger_dir + "postagger.sh"
model = postagger_dir + "models/english-left3words-distsim.tagger"

tmpfile = "tmp_build_postagging"

def main():
    for filename in sys.argv[1:]:
        print 'Processing', filename

        filename = os.path.abspath(filename)
        os.system(postagger_sh + " " + model + " " + filename + " > " + tmpfile)

        f_tmp = open(tmpfile, 'r')
        f_in = open(filename, 'w')
        f_out = open(filename + ".pos", 'w')

        for line in f_tmp:
            words = []
            postaggings = []
            for w in line.split(" "):
                w = w.split("_")
                words.append(w[0])
                postaggings.append(w[1])
            f_in.write(" ".join(words) + "\n")
            f_out.write(" ".join(postaggings))

        f_tmp.close()
        f_in.close()
        f_out.close()

        print 'Done'

    os.remove(tmpfile)


if __name__ == '__main__':
    main()
