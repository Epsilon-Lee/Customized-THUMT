src_path=IWSLT/test.en.tok.sub600.50times
tgt_path=IWSLT/pred/pred.test.top50.sub600.tok
output=IWSLT/pred/pred.test.top50.sub600.score
checkpoint=train_en2de_eval2k/eval
src_vocab=IWSLT/vocab.30k.en.txt
tgt_vocab=IWSLT/vocab.30k.de.txt
model=transformer
parameters="device_list=[0],eval_batch_size=64"

python thumt/bin/scorer.py \
    --input $src_path $tgt_path \
    --output $output \
    --checkpoint $checkpoint \
    --vocabulary $src_vocab $tgt_vocab \
    --model $model \
    --parameters $parameters \
