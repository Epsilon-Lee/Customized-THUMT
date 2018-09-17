train_src=IWSLT/train.de.singlebpe.tok
train_tgt=IWSLT/train.en.singlebpe.tok
vocab_src=IWSLT/vocab.singlebpe.de.txt
vocab_tgt=IWSLT/vocab.singlebpe.en.txt
dev_src=IWSLT/dev.de.singlebpe.tok
dev_tgt=IWSLT/dev.en.singlebpe.tok
parameters="batch_size=1024,device_list=[0,1],train_steps=200000,eval_steps=2000,num_encoder_layers=3,num_decoder_layers=3,hidden_size=512,filter_size=2048,num_heads=8,residual_dropout=0.5"
output=train_de2en_bz2048_singlebpe_small

CUDA_VISIBLE_DEVICES=1,3 python thumt/bin/trainer.py \
    --model transformer \
    --input $train_src $train_tgt \
    --vocabulary $vocab_src $vocab_tgt \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
