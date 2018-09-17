train_src=IWSLT/train.de.tok.sub50k
train_tgt=IWSLT/train.en.tok.sub50k
vocab_src=IWSLT/vocab.30k.de.sub50k.txt
vocab_tgt=IWSLT/vocab.30k.en.sub50k.txt
dev_src=IWSLT/dev.de.tok
dev_tgt=IWSLT/dev.en.tok
parameters="constant_batch_size=True,batch_size=64,device_list=[0,1],train_steps=200000,eval_steps=2000,hidden_size=512"
output=train_de2en_sub50k_rnn_bz64

CUDA_VISIBLE_DEVICES=0,1 python thumt/bin/trainer.py \
    --model rnnsearch \
    --input $train_src $train_tgt \
    --vocabulary $vocab_src $vocab_tgt \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
