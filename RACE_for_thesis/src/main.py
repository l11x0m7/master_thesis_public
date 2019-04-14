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
import cPickle as pkl
import nn_layers
import lasagne.layers as L
from nn_layers import QuerySliceLayer
from nn_layers import AttentionSumLayer
from nn_layers import GatedAttentionLayerWithQueryAttention

def gen_examples(x1, x2, x3, x4, x5, y, batch_size, use_sentence=False, sent_num=None, concat=False):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = utils.get_minibatches(len(x1), batch_size)
    all_ex = []
    if not use_sentence:
        x4 = x5
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_x3 = [x3[t * 4 + k] for t in minibatch for k in range(4)]
        if sent_num is not None:
            mb_x4 = [x4[t][k] if k < len(x4[t]) else [] for t in minibatch for k in range(sent_num)]
        else:
            mb_x4 = [x4[t] for t in minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
        mb_x2, mb_mask2 = utils.prepare_data(mb_x2)
        mb_x3, mb_mask3 = utils.prepare_data(mb_x3)
        if sent_num is not None:
            mb_x4, mb_mask4 = utils.prepare_data(mb_x4)
        else:
            mb_x4, mb_mask4 = utils.prepare_data2D(mb_x4)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4, mb_y))

    return all_ex


def build_fn(args, embeddings):
    """
        Build training and testing functions.
    """
    in_x1 = T.imatrix('x1')
    in_x2 = T.imatrix('x2')
    in_x3 = T.imatrix('x3')
    in_x4 = T.imatrix('x4')
    in_mask1 = T.matrix('mask1')
    in_mask2 = T.matrix('mask2')
    in_mask3 = T.matrix('mask3')
    in_mask4 = T.matrix('mask4')
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

    if args.use_sentence or args.use_key_sentence:
        l_in4 = lasagne.layers.InputLayer((None, None), in_x4)
        l_mask4 = lasagne.layers.InputLayer((None, None), in_mask4)
        l_emb4 = lasagne.layers.EmbeddingLayer(l_in4, args.vocab_size,
                                               args.embedding_size, W=l_emb1.W)

    args.rnn_output_size = args.hidden_size * 2 if args.bidir else args.hidden_size

    if not args.tune_embedding:
        l_emb1.params[l_emb1.W].remove('trainable')
        l_emb2.params[l_emb2.W].remove('trainable')
        l_emb3.params[l_emb3.W].remove('trainable')
        if args.use_sentence or args.use_key_sentence:
            l_emb4.params[l_emb4.W].remove('trainable')

    # my addition---sentence level attention
    if args.use_sentence or args.use_key_sentence:
        network_sent_d = nn_layers.stack_rnn(l_emb4, l_mask4, 1, args.hidden_size,
                                           grad_clipping=args.grad_clipping,
                                           dropout_rate=args.dropout_rate,
                                           only_return_final=False,
                                           bidir=args.bidir,
                                           name='sentence_doc',
                                           rnn_layer=args.rnn_layer)
        l_sent_mask4 = nn_layers.SqueezeLayer(l_mask4)
        l_sent_mask4 = lasagne.layers.ReshapeLayer(l_sent_mask4, (in_x1.shape[0], args.sent_num))
        sent_d_network = nn_layers.SelfAttention([network_sent_d, l_mask4], args.rnn_output_size, name='doc')
        sent_d_network = lasagne.layers.ReshapeLayer(sent_d_network, (in_x1.shape[0], args.sent_num, args.rnn_output_size))

        network_q = nn_layers.stack_rnn(l_emb2, l_mask2, args.num_layers, args.hidden_size,
                                           grad_clipping=args.grad_clipping,
                                           dropout_rate=args.dropout_rate,
                                           only_return_final=False,
                                           bidir=args.bidir,
                                           name='sentence_q',
                                           rnn_layer=args.rnn_layer)
        network_q = nn_layers.SelfAttention([network_q, l_mask2], args.rnn_output_size, name='q')

        network_o = nn_layers.stack_rnn(l_emb3, l_mask3, args.num_layers, args.hidden_size,
                                           grad_clipping=args.grad_clipping,
                                           dropout_rate=args.dropout_rate,
                                           only_return_final=False,
                                           bidir=args.bidir,
                                           name='sentence_o',
                                           rnn_layer=args.rnn_layer)
        network_o = nn_layers.SelfAttention([network_o, l_mask3], args.rnn_output_size, name='option')
        network_o = lasagne.layers.ReshapeLayer(network_o, (in_x1.shape[0], 4, args.rnn_output_size))
        for i in xrange(2):
            sent_d_network = nn_layers.GateWithQuery([sent_d_network, network_q])
            sent_d_network = nn_layers.stack_rnn(sent_d_network, l_sent_mask4, 1, args.hidden_size,
                                      grad_clipping=args.grad_clipping,
                                      dropout_rate=args.dropout_rate,
                                      only_return_final=False,
                                      bidir=args.bidir,
                                      name='sent_d_' + str(i),
                                      rnn_layer=args.rnn_layer)
        sent_att = nn_layers.BilinearAttentionMatLayer([sent_d_network, network_q], args.rnn_output_size,
                                                   mask_input=l_sent_mask4, name='sent_level')
        att = nn_layers.MultiplyLayer([sent_d_network, sent_att])
        # batch x 4
        network_sent = nn_layers.BilinearDotLayer([network_o, att], 
                                                   args.rnn_output_size, 
                                                   name='sent_level',
                                                   mask_input=None)

    # original part
    if args.model == "GA":
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
        # assert args.model is None
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
    if args.att_func == 'mlp':
        att = nn_layers.MLPAttentionLayer([network1, network2], args.rnn_output_size,
                                          mask_input=l_mask1)
    elif args.att_func == 'bilinear':
        att = nn_layers.BilinearAttentionLayer([network1, network2], args.rnn_output_size,
                                               mask_input=l_mask1, name='')
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
                                   only_return_final=True,
                                   bidir=args.bidir,
                                   name='o',
                                   rnn_layer=args.rnn_layer)
    network3 = lasagne.layers.ReshapeLayer(network3, (in_x1.shape[0], 4, args.rnn_output_size))
    if args.use_sentence or args.use_key_sentence:
        network3 = nn_layers.OptionGateLayer([network3, network_sent])
    network = nn_layers.BilinearDotLayer([network3, att], args.rnn_output_size, name='')

    # if args.use_sentence or args.use_key_sentence:
    #     network = nn_layers.CombineLayer([network, network_sent])


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
    network_sent = lasagne.layers.get_output(network_sent, deterministic=True)
    sent_att = lasagne.layers.get_output(sent_att, deterministic=True)
    test_prob = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = T.argmax(test_prob, axis=-1)
    acc = T.sum(T.eq(test_prediction, in_y))
    test_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_x4, in_mask4, in_y], [acc, test_prediction, network_sent, sent_att], on_unused_input='warn')

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
    train_fn = theano.function([in_x1, in_mask1, in_x2, in_mask2, in_x3, in_mask3, in_x4, in_mask4, in_y],
                               loss, updates=updates, on_unused_input='warn')

    return train_fn, test_fn, params, all_params


def eval_acc(test_fn, all_examples, args, word_dict_r):
    """
        Evaluate accuracy on `all_examples`.
    """
    acc = 0
    n_examples = 0
    prediction = []
    sentence_store = []
    sentence_att = []
    sentence_article = []
    sentence_question = []
    sentence_options = []
    sentence_answers = []
    for x1, mask1, x2, mask2, x3, mask3, x4, mask4, y in all_examples:
        tot_acc, pred, network_sent, sent_att = test_fn(x1, mask1, x2, mask2, x3, mask3, x4, mask4, y)
        acc += tot_acc
        prediction += pred.tolist()
        sentence_store += network_sent.tolist()
        sentence_att += sent_att.tolist()
        sentence_article += x4.reshape((x4.shape[0] // args.sent_num, args.sent_num, x4.shape[1])).tolist()
        sentence_question += x2.tolist()
        sentence_options += x3.reshape((len(x2), 4, x3.shape[1])).tolist()
        sentence_answers += y
        n_examples += len(x1)
        assert len(sentence_article) == len(sentence_question) == len(sentence_options) == len(sentence_answers)
    if args.test_only is True:
        with open('cache/visualization.pkl', 'w') as fw:
            sentence_article = [[[word_dict_r[w] for w in sent]
                                          for sent in sample]
                                          for sample in sentence_article]
            sentence_question = [[word_dict_r[w] for w in sent]
                                          for sent in sentence_question]
            sentence_options = [[[word_dict_r[w] for w in sent]
                                          for sent in sample]
                                          for sample in sentence_options]
            pkl.dump([sentence_store, sentence_att, 
                sentence_article, sentence_question, 
                sentence_options, sentence_answers, prediction], fw)


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
    # word_dict = pickle.load(open("../obj/dict.pkl", "rb"))
    logging.info('-' * 50)
    embeddings = utils.gen_embeddings(word_dict, args.embedding_size, args.embedding_file)
    (args.vocab_size, args.embedding_size) = embeddings.shape
    logging.info('Compile functions..')
    train_fn, test_fn, params, all_params = build_fn(args, embeddings)
    logging.info('Done.')
    logging.info('-' * 50)
    logging.info(args)

    logging.info('-' * 50)
    logging.info('Intial test..')
    dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_y = utils.vectorize(dev_examples, word_dict, sort_by_len=not args.test_only, concat=args.concat)
    word_dict_r = dict(zip(word_dict.values(), word_dict.keys()))
    word_dict_r[0] = '<PAD>'
    word_dict_r[1] = '<UNK>'
    assert len(dev_x1) == args.num_dev
    all_dev = gen_examples(dev_x1, dev_x2, dev_x3, dev_x4, dev_x5, dev_y, args.batch_size, args.use_sentence, args.sent_num, args.concat)
    dev_acc, pred = eval_acc(test_fn, all_dev, args, word_dict_r)
    logging.info('Dev accuracy: %.2f %%' % dev_acc)
    best_acc = dev_acc
    if args.test_only:
        return
    utils.save_params(args.model_file, all_params, epoch=0, n_updates=0)

    # Training
    logging.info('-' * 50)
    logging.info('Start training..')
    train_x1, train_x2, train_x3, train_x4, train_x5, train_y = utils.vectorize(train_examples, word_dict, concat=args.concat)
    assert len(train_x1) == args.num_train
    start_time = time.time()
    n_updates = 0

    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_x5, train_y, args.batch_size, args.use_sentence, args.sent_num, args.concat)
    for epoch in range(args.num_epoches):
        np.random.shuffle(all_train)
        for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4, mb_y) in enumerate(all_train):
            train_loss = train_fn(mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4, mb_y)
            if idx % 100 == 0:
                logging.info('#Examples = %d, max_len = %d' % (len(mb_x1), mb_x1.shape[1]))
                logging.info('Epoch = %d, iter = %d (max = %d), loss = %.2f, elapsed time = %.2f (s)' % (epoch, idx, len(all_train), train_loss, time.time() - start_time))
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                samples = sorted(np.random.choice(args.num_train, min(args.num_train, args.num_dev),
                                                  replace=False))
                sample_train = gen_examples([train_x1[k] for k in samples],
                                            [train_x2[k] for k in samples],
                                            [train_x3[k * 4 + o] for k in samples for o in range(4)],
                                            [train_x4[k] for k in samples],
                                            [train_x5[k] for k in samples],
                                            [train_y[k] for k in samples],
                                            args.batch_size, args.use_sentence, args.sent_num, args.concat)
                acc, pred = eval_acc(test_fn, sample_train, args, word_dict_r)
                logging.info('Train accuracy: %.2f %%' % acc)
                dev_acc, pred = eval_acc(test_fn, all_dev, args, word_dict_r)
                logging.info('Dev accuracy: %.2f %%' % dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    logging.info('Best dev accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%'
                                 % (epoch, n_updates, dev_acc))
                    utils.save_params(args.model_file, all_params, epoch=epoch, n_updates=n_updates)


def data_debug(args):
    # test the input data format
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(args.train_file, 100, relabeling=args.relabeling)
    word_dict = utils.build_dict(train_examples[0] + train_examples[1] + train_examples[2], args.max_vocab_size)
    train_x1, train_x2, train_x3, train_x4, train_x5, train_y = utils.vectorize(train_examples, word_dict, concat=args.concat)
    all_train = gen_examples(train_x1, train_x2, train_x3, train_x4, train_x5, train_y, args.batch_size, args.use_sentence, args.sent_num, args.concat)
    # np.random.shuffle(all_train)
    for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_x3, mb_mask3, mb_x4, mb_mask4, mb_y) in enumerate(all_train):
        print(mb_x1.shape, mb_x4.shape, mb_mask4.shape)
        print(mb_x4[0])
        print(mb_mask4[0])


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

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

    logging.info(' '.join(sys.argv))
    main(args)
    # data_debug(args)
