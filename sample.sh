input=IWSLT/test.en.tok.sub600
output=IWSLT/pred/pred.test.sample100.de.tok.sub600
python thumt/bin/sampler.py \
    --models transformer \
    --input $input \
    --output $output \
    --vocabulary IWSLT/vocab.30k.en.txt IWSLT/vocab.30k.de.txt \
    --checkpoints train_en2de_eval2k/eval \
    --parameters="device_list=[0,2],decode_batch_size=2,top_beams=1,beam_size=4" \
    #--verbose
