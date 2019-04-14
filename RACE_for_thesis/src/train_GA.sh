model_name='GA_glove300_key_sentlen30_with_ga_gating'
model='GA'
THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 nohup python2 main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.300d.txt -optimizer sgd -dropout_rate 0.5 -lr 0.3 -num_GA_layers 1 -hidden_size 128 -model ${model} -model_file ../obj/model_${model_name}.pkl.gz -debug False -sent_num 30 -use_sentence False -use_key_sentence True -tune_embedding True -eval_iter 1000 > log/${model_name}.log 2>&1 &
