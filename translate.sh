input=WMT14/newstest2014.bpe.en
output=Pred/newstest2014.bpe.de.bigmodel_baseline
CUDA_VISIBLE_DEVICES=1,3 python thumt/bin/translator.py \
    --models transformer \
    --input $input \
    --output $output \
    --vocabulary WMT14/vocab.en.bpe.txt WMT14/vocab.de.bpe.txt \
    --checkpoints ../thu_ckpts/WMT14-EN-DE/big_ckpt_eval \
    --parameters="device_list=[0,1],decode_batch_size=30,top_beams=1,beam_size=4" \
