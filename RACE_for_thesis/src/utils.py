import lasagne
import numpy as np
import config
import cPickle as pickle
import gzip
import logging
from collections import Counter
import os
import json
import os


def load_data(in_file, max_example=None, relabeling=True, question_belong=[]):
    documents = []
    sent_documents = []
    key_sent_documents = []
    questions = []
    answers = []
    options = []
    num_examples = 0
    def get_file(path):
        files = []
        for inf in os.listdir(path):
            new_path = os.path.join(path, inf)
            if os.path.isdir(new_path):
                assert inf in ["middle", "high"]
                files += get_file(new_path)
            else:
                if new_path.find(".DS_Store") != -1:
                    continue
                files += [new_path]
        return files
    files = get_file(in_file)
    for inf in files:
        obj = json.load(open(inf, "r"))
        for i, q in enumerate(obj["questions"]):
            question_belong += [inf + "_" + str(i)]
            documents += [obj["article"]]
            sent_documents += [obj['sent_article']]
            key_sent_documents.append(obj['key_sent'][i])
            questions += [q]
            assert len(obj["options"][i]) == 4
            options += obj["options"][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break
    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list
    documents = clean(documents)
    sent_documents = clean(sent_documents)
    questions = clean(questions)
    options = clean(options)
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options, sent_documents, key_sent_documents, answers)


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    word_count['a'] = 100000
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to pad
    # leave 1 to unk
    # leave 2 to @
    dic = {w[0]: index + 3 for (index, w) in enumerate(ls)}
    dic['@'] = 2
    return dic

def vectorize(examples, word_dict,
              sort_by_len=True, verbose=True, concat=False):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_x3 = []
    in_x4 = []
    in_x5 = []
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 1 for w in st]
        return seq

    for idx, (d, q, sd, ksd, a) in enumerate(zip(examples[0], examples[1], examples[3], examples[4], examples[5])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        # assert 0 <= a <= 3
        seq1 = get_vector(d_words)
        seq2 = get_vector(q_words)
        seq3 = []
        for sd_words in sd.split('|||'):
            seq3.append(get_vector(sd_words.split(' ')))
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
            in_x4 += [seq3]
            in_x5 += [np.asarray(seq3)[ksd].tolist() if len(ksd) != 0 else []]
            option_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + examples[2][i + idx * 4]
                else:
                    op = examples[2][i + idx * 4]
                op = op.split(' ')
                option = get_vector(op)
                assert len(option) > 0
                option_seq += [option]
            in_x3 += [option_seq]
            in_y.append(a)
        if verbose and (idx % 10000 == 0):
            logging.info('Vectorization: processed %d / %d' % (idx, len(examples[0])))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        in_x3 = [in_x3[i] for i in sorted_index]
        in_x4 = [in_x4[i] for i in sorted_index]
        in_x5 = [in_x5[i] for i in sorted_index]
    new_in_x3 = []
    for i in in_x3:
        #print i
        new_in_x3 += i
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, in_x4, in_x5, in_y


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype(np.int32)
    x_mask = np.zeros((n_samples, max_len)).astype(np.float32)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def prepare_data2D(seqs, sent_num):
    # seqs must be 2D level except the batch dimension.
    lengths = [[len(s) for s in seq] for seq in seqs]
    sent_lengths = [len(length) for length in seqs]
    n_samples = len(seqs)
    max_len_sent = np.max(sent_lengths)
    max_len_word = np.max([len(s) for seq in seqs for s in seq])
    if sent_num is None:
        sent_num = max_len_sent
    x = np.zeros((n_samples, sent_num, max_len_word)).astype(np.int32)
    x_mask = np.zeros((n_samples, sent_num, max_len_word)).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for idx2, s in enumerate(seq):
            x[idx, idx2, :min(lengths[idx][idx2], sent_num)] = s[:min(sent_num, len(s))]
            x_mask[idx, idx2, :min(lengths[idx][idx2], sent_num)] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None,
                   init=lasagne.init.Uniform()):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = max(word_dict.values()) + 1
    embeddings = init((num_words, dim))
    embeddings[0] = [0] * dim
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
                mu = embeddings[word_dict[sp[0]]].mean()
                #print embeddings[word_dict[sp[0]]]
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu += mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
    """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
    """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic
