model='GA'
model_name='../my_obj/model_new_GA_with_feat.pkl.gz'
emb_file='../data/embedding/glove.6B.100d.txt'
char_emb_file='../data/embedding/glove.840B.300d-char.txt'
embed_size=100
gpu=gpu1
option_suffix="-num_GA_layers 1 -hidden_size 128 -is_align True -max_d_len 500 -max_q_len 20 -max_o_len 15 -use_relu True -use_feat True -char_embedding_file ${char_emb_file}"

echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/dev -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model}  ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/middle -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/high -embedding_size ${embed_size} -pre_trained ${model_name} -test_only True -model ${model} ${option_suffix}
