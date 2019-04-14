model_name='SAR_glove300_key_sentlen30_with_ga_gating'
model='SAR'
THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 nohup python2 main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.300d.txt -model ${model} -optimizer sgd -lr 0.1 -dropout_rate 0.5 -model_file ../obj/model_${model_name}.pkl.gz -debug False -sent_num 30 -use_sentence False -use_key_sentence True -tune_embedding True > log/$model_name.log 2>&1 &
