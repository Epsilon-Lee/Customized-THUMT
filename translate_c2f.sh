input=WMT14/newstest2014.bpe.en
output=Pred/newstest2014.bpe.de.baseline_eval_29.77
CUDA_VISIBLE_DEVICES=1 python thumt/bin/translator_c2f.py \
    --models c2f-transformer \
    --submodels c2f-l4 \
    --input $input \
    --output $output \
    --vocabulary WMT14/vocab.en.bpe.txt WMT14/vocab.de.bpe.txt \
    --c2f-vocabulary WMT14/vocab.de.bpe.h22.txt \
    --checkpoints train_wmt_baseline_eval_29.77 \
    --parameters="device_list=[0],decode_batch_size=30,top_beams=1,beam_size=4" \
