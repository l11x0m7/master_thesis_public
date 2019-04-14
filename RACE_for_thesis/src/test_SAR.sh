gpu=gpu0
output_suffix=SAR_glove300_key_sentlen30_with_ga_gating
option_suffix='-sent_num 30 -use_sentence False -use_key_sentence True'
embed_size=300
model=../obj/model_${output_suffix}.pkl.gz
echo ${output_suffix}
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_size ${embed_size} -pre_trained ${model} -test_only True ${option_suffix} #-test_output test_${output_suffix}.txt ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_size ${embed_size} -pre_trained ${model} -test_only True ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test/middle -embedding_size ${embed_size}  -pre_trained ${model} -test_only True ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python2 main.py -train_file ../data/data/train -dev_file ../data/data/test/high -embedding_size ${embed_size} -pre_trained ${model} -test_only True ${option_suffix}
