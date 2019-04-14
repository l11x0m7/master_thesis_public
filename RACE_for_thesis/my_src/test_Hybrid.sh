output_suffix=Hybrid
model=../my_obj/model_new_Hybrid_glove300.pkl.gz
# model=../my_obj/model_Hybrid.pkl.gz
embed_size=300
gpu=gpu1
option_suffix="-is_align True -max_d_len 500 -max_q_len 20 -max_o_len 15 -grad_clipping 10.0 -eval_iter 1000 -use_relu True"
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model Hybrid ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/dev -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model Hybrid ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/middle -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model Hybrid ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/my_data/train -dev_file ../data/my_data/test/high -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model Hybrid ${option_suffix}

