train_src=IWSLT/train.de.tok
train_tgt=IWSLT/train.en.tok
train_tgt_l1=IWSLT/train.en.tok.h21
train_tgt_l0=IWSLT/train.en.tok.h8
vocab_src=IWSLT/vocab.30k.de.txt
vocab_tgt=IWSLT/vocab.30k.en.txt
vocab_tgt_l1=IWSLT/vocab.en.h21.txt
vocab_tgt_l0=IWSLT/vocab.en.h8.txt
dev_src=IWSLT/dev.de.tok
dev_tgt=IWSLT/dev.en.tok
parameters="batch_size=2048,device_list=[0],train_steps=200000,eval_steps=200,num_encoder_layers=6,num_decoder_layers=6,hidden_size=512,filter_size=2048,num_heads=8,residual_dropout=0.2"
output=train_de2en_bz2048_prune_l1_v0_h8_21_drop0.2

CUDA_VISIBLE_DEVICES=1 python thumt/bin/trainer_prune_l01.py \
    --model prune-l1-transformer \
    --submodel prune-l1-v0 \
    --input $train_src $train_tgt \
    --c2f-input $train_tgt_l0 $train_tgt_l1 \
    --vocabulary $vocab_src $vocab_tgt \
    --c2f-vocabulary $vocab_tgt_l0 $vocab_tgt_l1 \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
