output_suffix=GA_glove300_key_sentlen30_with_gating_visual
model=../obj/model_${output_suffix}.pkl.gz
embed_size=300
gpu=gpu1
option_suffix="-num_GA_layers 1 -hidden_size 128 -sent_num 30 -use_sentence False -use_key_sentence True"
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test/middle -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model GA ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test/high -embedding_size ${embed_size} -pre_trained ${model} -test_only True -model GA ${option_suffix}

