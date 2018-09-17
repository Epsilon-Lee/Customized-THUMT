hier_list="6 7 9 10 12 13 14 15"
for hier in $hier_list; do
    python thumt/scripts/build_vocab.py WMT14/corpus.bpe32k.de.neural.h$hier WMT14/vocab.de.bpe.neural.h$hier
done
