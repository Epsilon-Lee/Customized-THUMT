train_src=IWSLT/train.de.tok
train_tgt=IWSLT/train.en.tok
train_tgt_l0=IWSLT/train.en.tok.h21
vocab_src=IWSLT/vocab.30k.de.txt
vocab_tgt=IWSLT/vocab.30k.en.txt
vocab_tgt_l0=IWSLT/vocab.en.h21.txt
vocab_map_0=IWSLT/bc-5000.code.h21.h21-token
vocab_map_1=IWSLT/bc-5000.code.h21.word-token
dev_src=IWSLT/dev.de.tok
dev_tgt=IWSLT/dev.en.tok
parameters="batch_size=1024,device_list=[0,1],train_steps=200000,eval_steps=20,num_encoder_layers=6,num_decoder_layers=6,hidden_size=512,filter_size=2048,num_heads=8,residual_dropout=0.5"
output=train_de2en_bz2048_prune_l0_h21_regemb

CUDA_VISIBLE_DEVICES=1,2 python thumt/bin/trainer_c2f_regemb.py \
    --model prune-l0-transformer \
    --submodel prune-l0-regemb \
    --input $train_src $train_tgt \
    --c2f-input $train_tgt_l0  \
    --vocabulary $vocab_src $vocab_tgt \
    --c2f-vocabulary $vocab_tgt_l0  \
    --c2f-vocabmap $vocab_map_0 $vocab_map_1 \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
