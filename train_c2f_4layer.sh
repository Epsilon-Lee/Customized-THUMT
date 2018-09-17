export PYTHONPATH=$PYTHONPATH:"/app/home/"

#train_src=WMT14/corpus.bpe32k.en
#train_tgt=WMT14/corpus.bpe32k.de
#train_tgt_l1=WMT14/corpus.bpe32k.de.neural.h5
#train_tgt_l2=WMT14/corpus.bpe32k.de.neural.h8
#train_tgt_l3=WMT14/corpus.bpe32k.de.neural.h11
#train_tgt_l4=WMT14/corpus.bpe32k.de.neural.h20
#vocab_src=WMT14/vocab.en.bpe.txt
#vocab_tgt=WMT14/vocab.de.bpe.txt
#vocab_tgt_l1=WMT14/vocab.de.bpe.neural.h5.txt
#vocab_tgt_l2=WMT14/vocab.de.bpe.neural.h8.txt
#vocab_tgt_l3=WMT14/vocab.de.bpe.neural.h11.txt
#vocab_tgt_l4=WMT14/vocab.de.bpe.neural.h20.txt
#dev_src=WMT14/newstest2013.bpe.en
#dev_tgt=WMT14/newstest2013.bpe.de

train_src=WMT14/corpus.bpe32k.en
train_tgt=WMT14/corpus.bpe32k.de
train_tgt_l1=WMT14/corpus.bpe32k.de.h5
train_tgt_l2=WMT14/corpus.bpe32k.de.h7
train_tgt_l3=WMT14/corpus.bpe32k.de.h10
train_tgt_l4=WMT14/corpus.bpe32k.de.h22
vocab_src=WMT14/vocab.en.bpe.txt
vocab_tgt=WMT14/vocab.de.bpe.txt
vocab_tgt_l1=WMT14/vocab.de.bpe.h5.txt
vocab_tgt_l2=WMT14/vocab.de.bpe.h7.txt
vocab_tgt_l3=WMT14/vocab.de.bpe.h10.txt
vocab_tgt_l4=WMT14/vocab.de.bpe.h22.txt
dev_src=WMT14/newstest2013.bpe.en
dev_tgt=WMT14/newstest2013.bpe.de
parameters="batch_size=4096,device_list=[0,1,2,3],train_steps=500000,eval_steps=2000,num_encoder_layers=6,num_decoder_layers=6,hidden_size=512,filter_size=2048,num_heads=8,attention_dropout=0.1,relu_dropout=0.1,residual_dropout=0.1,update_cycle=2"
output=train_wmt_c2f_4layers_softcoherency

#CUDA_VISIBLE_DEVICES=3 
python thumt/bin/trainer_4layers.py \
    --model c2f-4l-transformer \
    --submodel c2f-4l \
    --input $train_src $train_tgt \
    --c2f-input $train_tgt_l1 $train_tgt_l2 $train_tgt_l3 $train_tgt_l4 \
    --vocabulary $vocab_src $vocab_tgt \
    --c2f-vocabulary $vocab_tgt_l1 $vocab_tgt_l2 $vocab_tgt_l3 $vocab_tgt_l4  \
    --validation $dev_src \
    --references $dev_tgt \
    --output $output \
    --parameters=$parameters \
    --checkpoint ./train_wmt_baseline_eval \
#    --checkpoint ../thu_ckpts/ckpt_c2f/train_nist_c2f_h21_eval \
