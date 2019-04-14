import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle
import sys
import time
import utils
import config
import logging
import nn_layers
import lasagne.layers as L
from nn_layers import QuerySliceLayer, CharEncoding
from nn_layers import AttentionSumLayer
from nn_layers import GatedAttentionLayerWithQueryAttention
from nn_layers import BiDirectionAttentionLayer, HybridInteractionLayer
from nn_layers import SelfAttention, HybridInteractionWithDense

def gen_examples(X, args):
    """
        Divide examples into batches of size `batch_size`.
    """
    x1, x2, x3, y = X
    minibatches = utils.get_minibatches(len(x1), args.batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_y = [y[t] for t in minibatch]
        # if args.is_align:
        #     max_d_len = args.max_d_len
        #     max_q_len = args.max_q_len
        #     max_o_len = args.max_o_len
        # else:
        max_d_len = args.max_d_len
        max_q_len = args.max_q_len
        max_o_len = None
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1, max_d_len, dynamic=True)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2, max_q_len, dynamic=True)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3, max_o_len, dynamic=True)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y))
    return all_ex


def gen_examples_all(X, args):
    """
        Divide examples into batches of size `batch_size`.
    """
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y = X
    minibatches = utils.get_minibatches(len(x1), args.batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x4 = [x4[t] for t in minibatch]
        mb_x5 = [x5[t] for t in minibatch]
        mb_x6 = [x6[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x7 = [x7[t] for t in minibatch]
        mb_x8 = [x8[t] for t in minibatch]
        mb_x9 = [x9[t * 4 + k] for t in minibatch for k in range(4)]
        mb_x10 = [x10[t] for t in minibatch]
        mb_y = [y[t] for t in minibatch]
        # if args.is_align:
        #     max_d_len = args.max_d_len
        #     max_q_len = args.max_q_len
        #     max_o_len = args.max_o_len
        # else:
        max_d_len = args.max_d_len
        max_q_len = args.max_q_len
        max_o_len = None
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1, max_d_len, dynamic=True)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2, max_q_len, dynamic=True)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3, max_o_len, dynamic=True)
        mb_x4 = utils.prepare_data_char(mb_x4, args, max_d_len, dynamic=True, mask=False)
        mb_x5 = utils.prepare_data_char(mb_x5, args, max_q_len, dynamic=True, mask=False)
        mb_x6 = utils.prepare_data_char(mb_x6, args, max_o_len, dynamic=True, mask=False)
        mb_x7 = utils.prepare_data(mb_x7, max_d_len, dynamic=True, mask=False)
        mb_x8 = utils.prepare_data(mb_x8, max_q_len, dynamic=True, mask=False)
        mb_x9 = utils.prepare_data(mb_x9, max_o_len, dynamic=True, mask=False)
        mb_x10 = utils.prepare_data(mb_x10, mb_x1.shape[1], dynamic=True, mask=False)
        all_ex.append((mb_x1, mb_mask1, 
                       mb_x2, mb_mask2, 
                       mb_x3, mb_mask3, 
                       mb_x4, mb_x5,
                       mb_x6, mb_x7,
                       mb_x8, mb_x9,
                       mb_x10, mb_y))
    return all_ex


def build_fn(args, embeddings, char_embeddings=None):
    """
        Build training and testing functions.
    """
    # in_x1: doc words
    # in_x2: question words
    # in_x3: option words
    # in_x4: doc chars
    # in_x5: question chars
    # in_x6: option chars
    # in_x7: doc ners
    # in_x8: question ners
    # in_x9: option ners
    # in_x10: co features
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_x3 = T.imatrix('x3')
    in_x4 = T.itensor3('x4')
    in_x5 = T.itensor3('x5')
    in_x6 = T.itensor3('x6')
    in_x7 = T.imatrix('x7')
    in_x8 = T.imatrix('x8')
    in_x9 = T.imatrix('x9')
    in_x10 = T.imatrix('x10')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_mask3 = T.matrix('mask3')
    in_y = T.ivector('y')

    l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
    l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, args.vocab_size,
                                           args.embedding_size, W=embeddings)

    l_in2 = lasagne.layers.InputLayer((None, None), in_x2)
    l_mask2 = lasagne.layers.InputLayer((None, None), in_mask2)
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    l_in3 = lasagne.layers.InputLayer((None, None), in_x3)
    l_mask3 = lasagne.layers.InputLayer((None, None), in_mask3)
    l_emb3 = lasagne.layers.EmbeddingLayer(l_in3, args.vocab_size,
                                           args.embedding_size, W=l_emb1.W)

    if not args.tune_embedding:
        l_emb1.params[l_emb1.W].remove('trainable')
        l_emb2.params[l_emb2.W].remove('trainable')
        l_emb3.params[l_emb3.W].remove('trainable')

    if args.use_feat:
        W = lasagne.init.Normal() if char_embeddings is None else char_embeddings
        l_in4 = lasagne.layers.InputLayer((None, None, args.max_word_len), in_x4)
        l_emb4 = lasagne.layers.EmbeddingLayer(l_in4, args.char_vocab_size,
                                               args.char_embedding_size, W=W)
        l_in5 = lasagne.layers.InputLayer((None, None, args.max_word_len), in_x5)
        l_emb5 = lasagne.layers.EmbeddingLayer(l_in5, args.char_vocab_size,
                                               args.char_embedding_size, W=W)
        l_in6 = lasagne.layers.InputLayer((None, None, args.max_word_len), in_x6)
        l_emb6 = lasagne.layers.EmbeddingLayer(l_in6, args.char_vocab_size,
                                               args.char_embedding_size, W=W)
        l_in7 = lasagne.layers.InputLayer((None, None), in_x7)
        l_emb7 = lasagne.layers.EmbeddingLayer(l_in7, args.ner_vocab_size,
                                               args.ner_embedding_size)
        l_in8 = lasagne.layers.InputLayer((None, None), in_x8)
        l_emb8 = lasagne.layers.EmbeddingLayer(l_in8, args.ner_vocab_size,
                                               args.ner_embedding_size)
        l_in9 = lasagne.layers.InputLayer((None, None), in_x9)
        l_emb9 = lasagne.layers.EmbeddingLayer(l_in9, args.ner_vocab_size,
                                               args.ner_embedding_size)
        l_in10 = lasagne.layers.InputLayer((None, None), in_x10)
        l_emb10 = lasagne.layers.EmbeddingLayer(l_in10, 2,
                                               args.co_feat_embedding_size)
        l_emb_out4 = CharEncoding(l_emb4, num_filters=32, filter_size=2)
        l_emb_out5 = CharEncoding(l_emb5, num_filters=32, filter_size=2, W=l_emb_out4.W, b=l_emb_out4.b)
        l_emb_out6 = CharEncoding(l_emb6, num_filters=32, filter_size=2, W=l_emb_out4.W, b=l_emb_out4.b)
        l_emb1 = lasagne.layers.ConcatLayer([l_emb1, l_emb_out4, l_emb7, l_emb10], axis=-1)
        l_emb2 = lasagne.layers.ConcatLayer([l_emb2, l_emb_out5, l_emb8], axis=-1)
        l_emb3 = lasagne.layers.ConcatLayer([l_emb3, l_emb_out6, l_emb9], axis=-1)


    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size

    # Now containing BiDAF
    if args.model == 'BiDAF':
        logging.info('Using the BiDAF model')
        l_d = nn_layers.stack_rnn(l_emb1, l_mask1, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='d',
                                       rnn_layer=args.rnn_layer)

        l_q = nn_layers.stack_rnn(l_emb2, l_mask2, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)

        q_length = nn_layers.LengthLayer(l_mask2)
        network2 = QuerySliceLayer([l_q, q_length])
        
        # B * D * 8h
        bi_att_out= BiDirectionAttentionLayer([l_d, l_q, l_mask1, l_mask2], args.rnn_output_size)
        # B * D * 2h
        bi_att_out = nn_layers.stack_rnn(bi_att_out, l_mask1, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='att_d',
                                       rnn_layer=args.rnn_layer)
        network1 = bi_att_out
    elif args.model == "Hybrid":
        logging.info('Using the Hybrid model')
        l_d = nn_layers.stack_rnn(l_emb1, l_mask1, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='d',
                                       rnn_layer=args.rnn_layer)

        l_q = nn_layers.stack_rnn(l_emb2, l_mask2, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)

        # q_length = nn_layers.LengthLayer(l_mask2)
        network2 = SelfAttention([l_q, l_mask2], args.rnn_output_size, name='self_q')
        
        # B * D * 4h
        hybrid_out = HybridInteractionLayer([l_d, l_q, l_mask1, l_mask2], 
                                            args.rnn_output_size, 
                                            name='hybrid1', 
                                            use_relu=args.use_relu)
        # B * D * 2h
        if not args.use_relu:
            hybrid_out = nn_layers.stack_rnn(hybrid_out, l_mask1, 1, args.hidden_size,
                                           grad_clipping=args.grad_clipping,
                                           dropout_rate=args.dropout_rate,
                                           only_return_final=False,
                                           bidir=args.bidir,
                                           name='att_d',
                                           rnn_layer=args.rnn_layer)
        network1 = hybrid_out
    elif args.model == 'Conv':
        logging.info('Using the Convolution model')
        l_emb1 = lasagne.layers.DropoutLayer(l_emb1, p=args.dropout_rate)
        l_emb2 = lasagne.layers.DropoutLayer(l_emb2, p=args.dropout_rate)
        l_d = nn_layers.ConvLayer(l_emb1, l_mask1, args.rnn_output_size, args, depth=[3, 5, 7, 9])

        l_q = nn_layers.ConvLayer(l_emb2, l_mask2, args.rnn_output_size, args, depth=[])
        
        # q_length = nn_layers.LengthLayer(l_mask2)
        # network2 = SelfAttention([l_q, l_mask2], args.rnn_output_size, name='self_q')
        network2 = l_q
        # l_d = GatedAttentionLayerWithQueryAttention([l_d, l_q, l_mask2])
        network1 = l_d
    elif args.model == "GA":
        logging.info('Using the GA model')
        l_d = l_emb1
        # NOTE: This implementation slightly differs from the original GA reader. Specifically:
        # 1. The query GRU is shared across hops.
        # 2. Dropout is applied to all hops (including the initial hop).
        # 3. Gated-attention is applied at the final layer as well.
        # 4. No character-level embeddings are used.

        l_q = nn_layers.stack_rnn(l_emb2, l_mask2, 1, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
        q_length = nn_layers.LengthLayer(l_mask2)
        network2 = QuerySliceLayer([l_q, q_length])
        for layer_num in xrange(args.num_GA_layers):
            l_d = nn_layers.stack_rnn(l_d, l_mask1, 1, args.hidden_size,
                                      grad_clipping=args.grad_clipping,
                                      dropout_rate=args.dropout_rate,
                                      only_return_final=False,
                                      bidir=args.bidir,
                                      name='d' + str(layer_num),
                                      rnn_layer=args.rnn_layer)
            l_d = GatedAttentionLayerWithQueryAttention([l_d, l_q, l_mask2])
        network1 = l_d
    else:
        logging.info('Using the SAR model')
        assert args.model is None
        network1 = nn_layers.stack_rnn(l_emb1, l_mask1, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=(args.att_func == 'last'),
                                       bidir=args.bidir,
                                       name='d',
                                       rnn_layer=args.rnn_layer)

        network2 = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=True,
                                       bidir=args.bidir,
                                       name='q',
                                       rnn_layer=args.rnn_layer)
    if args.model == 'Conv':
        # B * P * D
        att = nn_layers.BilinearAttentionPQLayer([l_d, l_q, l_mask1, l_mask2], args.rnn_output_size)
        network3 = nn_layers.ConvLayer(l_emb3, l_mask3, args.rnn_output_size, args, depth=[])
    else:
        if args.att_func == 'mlp':
            att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                              mask_input=l_mask1)
        elif args.att_func == 'bilinear':
            att = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                                   mask_input=l_mask1)
        elif args.att_func == 'avg':
            att = nn_layers.AveragePoolingLayer(network1, mask_input=l_mask1)
        elif args.att_func == 'last':
            att = network1
        elif args.att_func == 'dot':
            att = nn_layers.DotProductAttentionLayer([network1, network2], mask_input=l_mask1)
        else:
            raise NotImplementedError('att_func = %s' % args.att_func)
        network3 = nn_layers.stack_rnn(l_emb3, l_mask3, args.num_layers, args.hidden_size,
                                       grad_clipping=args.grad_clipping,
                                       dropout_rate=args.dropout_rate,
                                       only_return_final=False,
                                       bidir=args.bidir,
                                       name='o',
                                       rnn_layer=args.rnn_layer)
    if args.model == 'Conv':
        # B * 4
        network = nn_layers.BilinearAttentionPALayer([network3, att, l_mask3, l_mask1], args.rnn_output_size)
    else:
        # choose one of the followings
        # 1. self attention: (B*4, O, 2D) -> (B*4, 2D)
        # network3 = SelfAttention([network3, l_mask3], args.rnn_output_size, name='self_o')
        # 2. interaction
        # (B, 4*O, 2D)
        if args.model == 'Hybrid':
            network3 = HybridInteractionWithDense([network3, network1, l_mask3, l_mask1], args.rnn_output_size, name='hybrid2')
        # (B*4, 2D)
        network3 = SelfAttention([network3, l_mask3], args.rnn_output_size, name='self_o')

        network3 = lasagne.layers.ReshapeLayer(network3, (in_x1.shape[0], 4, args.rnn_output_size))
        # B * 4
        network = nn_layers.BilinearDotLayer([network3, att], args.rnn_output_size)


    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        lasagne.layers.set_all_param_values(network, dic['params'])
        del dic['params']
        logging.info('Loaded pre-trained model: %s' % args.pre_trained)
        for dic_param in dic.iteritems():
            logging.info(dic_param)

    logging.info('#params: %d' % lasagne.layers.count_params(network, trainable=True))
    logging.info('#fixed params: %d' % lasagne.layers.count_params(network, trainable=False))
    for layer in lasagne.layers.get_all_layers(network):
        logging.info(layer)

    # Test functions
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    if args.use_feat:
        test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_x4, in_x5, in_x6, in_x7, in_x8, in_x9, in_x10, in_y], [acc, test_prediction], on_unused_input='warn')
    else:
        test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y], [acc, test_prediction], on_unused_input='warn')

    # Train functions
    train_prediction = lasagne.layers.get_output(network)
    train_prediction = T.clip(train_prediction, 1e-7, 1.0 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, in_y).mean()
    # TODO: lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    params = lasagne.layers.get_all_params(network)#, trainable=True)
    all_params = lasagne.layers.get_all_params(network)
    if args.optimizer == 'sgd':
        updates = lasagne.updates.sgd(loss, params, args.learning_rate)
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=args.learning_rate)
    else:
        raise NotImplementedError('optimizer = %s' % args.optimizer)
    if args.use_feat:
        train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_x4, in_x5, in_x6, in_x7, in_x8, in_x9, in_x10, in_y],
                               loss, updates=updates, on_unused_input='warn')
    else:
        train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_y],
                               loss, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, params, all_params


def eval_acc(test_fn, all_examples):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    prediction = []
    for x1, mask1, x2, mask2, x3, mask3, y in all_examples:
        tot_acc, pred = test_fn(x1, mask1, x2, mask2, x3, mask3, y)
        acc += tot_acc
        prediction += pred.tolist()
        n_examples += len(x1)
    return acc * 100.0 / n_examples, prediction

def eval_acc_all(test_fn, all_examples):
    """
        Evaluate accuracy n_exampleson `all_examples`.
    """
    acc = 0
    n_examples = 0
    prediction = []
    for x1, mask1, x2, mask2, x3, mask3, x4, x5, x6, x7, x8, x9, x10, y in all_examples:
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x6.shape)
        # print(x7.shape)
        # print(x8.shape)
        # print(x9.shape)
        # print(x10.shape)
        tot_acc, pred = test_fn(x1, mask1, x2, mask2, x3, mask3, x4, x5, x6, x7, x8, x9, x10, y)
        acc += tot_acc
        prediction += pred.tolist()
        n_examples += len(x1)
    return acc * 100.0 / n_examples, prediction


def main(args):
    logging.info('-' * 50)
    logging.info('Load data files..')
    question_belong = []
    if args.debug:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, 100, relabeling=args.relabeling, question_belong=question_belong)
    else:
        logging.info('*' * 10 + ' Train')
        train_examples = utils.load_data(args.train_file, relabeling=args.relabeling)
        logging.info('*' * 10 + ' Dev')
        dev_examples = utils.load_data(args.dev_file, args.max_dev, relabeling=args.relabeling, question_belong=question_belong)

    args.num_train = len(train_examples[0])
    args.num_dev = len(dev_examples[0])

    logging.info('-' * 50)
    logging.info('Build dictionary..')
    word_dict = utils.build_dict(train_examples[0] + train_examples[1] + train_examples[2], args.max_vocab_size)
    if args.use_feat:
        char_dict = utils.build_dict(train_examples[4] + train_examples[5] + train_examples[6], is_char=True)
        ner_dict = utils.build_dict(train_examples[7] + train_examples[8] + train_examples[9])
        args.char_vocab_size = max(char_dict.values()) + 1
        args.ner_vocab_size = max(ner_dict.values()) + 1
        if args.char_embedding_file:
            char_embeddings = utils.gen_embeddings(char_dict, args.char_embedding_size, args.char_embedding_file)
            args.char_vocab_size, args.char_embedding_size = char_embeddings.shape
        else:
            char_embeddings = None
    else:
        char_embeddings = None
    # word_dict = pickle.load(open("../obj/dict.pkl", "rb"))
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, params, all_params = build_fn(args, embeddings, char_embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    if args.use_feat:
        dev_x_y = utils.vectorize_all(dev_examples, word_dict, char_dict, ner_dict, sort_by_len=not args.test_only, concat=args.concat)
        all_dev = gen_examples_all(dev_x_y, args)
    else:
        dev_x_y = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
        all_dev = gen_examples(dev_x_y, args)

    assert len(dev_x_y[0]) == args.num_dev
    if args.use_feat:
        dev_acc, pred = eval_acc_all(test_fn, all_dev)
    else:
        dev_acc, pred = eval_acc(test_fn, all_dev)
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc
    if args.test_only:
        return
    utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)

    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    if args.use_feat:
        train_x_y = utils.vectorize_all(train_examples, word_dict, char_dict, ner_dict, concat=args.concat)
    else:
        train_x_y = utils.vectorize(train_examples, word_dict, concat=args.concat)
    assert len(train_x_y[0]) == args.num_train
    start_time = time.time()
    n_updates = 0

    if args.use_feat:
        all_train = gen_examples_all(train_x_y, args)
        for epoch in range(args.num_epoches):
            np.random.shuffle(all_train)
            for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, 
                      mb_x3, mb_mask3, mb_x4, mb_x5, 
                      mb_x6, mb_x7, mb_x8, mb_x9, 
                      mb_x10, mb_y) in enumerate(all_train):
                train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, 
                                      mb_x3, mb_mask3, mb_x4, mb_x5, 
                                      mb_x6, mb_x7, mb_x8, mb_x9, mb_x10, mb_y)
                if idx % 100 == 0:
                    logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                    logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
                n_updates += 1

                if n_updates % args.eval_iter == 0:
                    samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                      replace=False))
                    sample_train = gen_examples_all([[train_x_y[0][k] for k in samples],
                                                [train_x_y[1][k] for k in samples],
                                                [train_x_y[2][k * 4 + o] for k in samples for o in range(4)],
                                                [train_x_y[3][k] for k in samples],
                                                [train_x_y[4][k] for k in samples],
                                                [train_x_y[5][k * 4 + o] for k in samples for o in range(4)],
                                                [train_x_y[6][k] for k in samples],
                                                [train_x_y[7][k] for k in samples],
                                                [train_x_y[8][k * 4 + o] for k in samples for o in range(4)],
                                                [train_x_y[9][k] for k in samples],
                                                [train_x_y[10][k] for k in samples]],
                                                args)
                    acc, pred = eval_acc_all(test_fn, sample_train)
                    logging.info('Train accuracy: %.2f %%' % acc)
                    dev_acc, pred = eval_acc_all(test_fn, all_dev)
                    logging.info('Dev accuracy: %.2f %%' % dev_acc)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                     % (epoch, n_updates, dev_acc))
                        utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates)

    else:
        all_train = gen_examples(train_x_y, args)
        for epoch in range(args.num_epoches):
            np.random.shuffle(all_train)
            for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y) in enumerate(all_train):
                train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_y)
                if idx % 100 == 0:
                    logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                    logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
                n_updates += 1

                if n_updates % args.eval_iter == 0:
                    samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                      replace=False))
                    sample_train = gen_examples([[train_x_y[0][k] for k in samples],
                                                [train_x_y[1][k] for k in samples],
                                                [train_x_y[2][k * 4 + o] for k in samples for o in range(4)],
                                                [train_x_y[3][k] for k in samples]],
                                                args)
                    acc, pred = eval_acc(test_fn, sample_train)
                    logging.info('Train accuracy: %.2f %%' % acc)
                    dev_acc, pred = eval_acc(test_fn, all_dev)
                    logging.info('Dev accuracy: %.2f %%' % dev_acc)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                     % (epoch, n_updates, dev_acc))
                        utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates)


if __name__ == '__main__':
    args = config.get_args()
    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.train_file is None:
        raise ValueError('train_file is not specified.')

    if args.dev_file is None:
        raise ValueError('dev_file is not specified.')

    if args.rnn_type == 'lstm':
        args.rnn_layer = lasagne.layers.LSTMLayer
    elif args.rnn_type == 'gru':
        args.rnn_layer = lasagne.layers.GRULayer
    else:
        raise NotImplementedError('rnn_type = %s' % args.rnn_type)

    if args.embedding_file is not None:
        dim = utils.get_dim(args.embedding_file)
        if (args.embedding_size is not None) and (args.embedding_size != dim):
            raise ValueError('embedding_size = %d, but %s has %d dims.' %
                             (args.embedding_size, args.embedding_file, dim))
        args.embedding_size = dim
    elif args.embedding_size is None:
        raise RuntimeError('Either embedding_file or embedding_size needs to be specified.')

    if args.char_embedding_file is not None:
        dim = utils.get_dim(args.char_embedding_file)
        args.char_embedding_size = dim

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
