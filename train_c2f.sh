export PYTHONPATH=$PYTHONPATH:"/app/home/"

train_src=WMT14/corpus.bpe32k.en
train_tgt=WMT14/corpus.bpe32k.de
train_tgt_l0=WMT14/corpus.bpe32k.de.h22
vocab_src=WMT14/vocab.en.bpe.txt
vocab_tgt=WMT14/vocab.de.bpe.txt
vocab_tgt_l0=WMT14/vocab.de.bpe.h22.txt
dev_src=WMT14/newstest2013.bpe.en
dev_tgt=WMT14/newstest2013.bpe.de
parameters="batch_size=4096,device_list=[0,1,2,3],train_steps=500000,eval_steps=2000,num_encoder_layers=6,num_decoder_layers=6,hidden_size=512,filter_size=2048,num_heads=8,attention_dropout=0.1,residual_dropout=0.1,update_cycle=2"
output=train_wmt14_c2f_h22_pretrained

#CUDA_VISIBLE_DEVICES=3 
python thumt/bin/trainer_h8_11_15_21.py \
    --model c2f-transformer \
    --submodel c2f-l4 \
    --input $train_src $train_tgt \
    --c2f-input $train_tgt_l0  \
    --vocabulary $vocab_src $vocab_tgt \
    --c2f-vocabulary $vocab_tgt_l0  \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
    --checkpoint ./train_wmt_baseline_eval \
