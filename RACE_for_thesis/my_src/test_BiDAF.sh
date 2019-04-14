model='BiDAF'
model_name='../my_obj/model_new_BiDAF_glove300.pkl.gz'
emb_file='../data/embedding/glove.840B.300d.txt'
char_emb_file='../data/embedding/glove.840B.300d-char.txt'
embed_size=300
gpu=gpu1
option_suffix="-hidden_size 128 -is_align True -max_d_len 500 -max_q_len 20 -max_o_len 15"
# THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" stdbuf -i0 -e0 -o0 nohup python2 main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -dropout_rate 0.5 -lr 0.3 -num_GA_layers 1 -hidden_size 128 -model ${model} -model_file ../my_obj/model_${model_name}.pkl.gz -debug False -is_align True -max_d_len 500 -max_q_len 20 -max_o_len 15 > log/${model_name}.log 2>&1 &


echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/dev -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model}  ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/middle -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/high -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
