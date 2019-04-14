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
    questions = []
    answers = []
    options = []
    doc_ners = []
    ques_ners = []
    opt_ners = []
    doc_char = []
    ques_char = []
    opt_char = []
    document_co_occur_features = []
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
            document_co_occur_features += [obj['article_cooccur'][i]]
            doc_ners += [obj['article_ner']]
            doc_char += [obj['article_char']]
            questions += [q]
            ques_ners += [obj['questions_ner'][i]]
            ques_char += [obj['questions_char'][i]]
            assert len(obj["options"][i]) == 4
            options += obj["options"][i]
            opt_ners += obj['options_ner'][i]
            opt_char += obj['options_char'][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            num_examples += 1
        if (max_example is not None) and (num_examples >= max_example):
            break
    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list
    documents = clean(documents)
    questions = clean(questions)
    options = clean(options)
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, options, answers, doc_char, ques_char, opt_char, doc_ners, ques_ners, opt_ners, document_co_occur_features)


def build_dict(sentences, max_words=None, is_char=False):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    word_count['a'] = 100000
    for sent in sentences:
        for w in sent.split(' '):
            if is_char:
                for c in w:
                    word_count[c] += 1
            else:
                word_count[w] += 1

    if max_words:
        ls = word_count.most_common(max_words)
    else:
        ls = word_count.most_common()
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
    in_y = []
    def get_vector(st):
        seq = [word_dict[w] if w in word_dict else 1 for w in st]
        return seq

    for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[3])):
        d_words = d.split(' ')
        q_words = q.split(' ')
        # assert 0 <= a <= 3
        seq1 = get_vector(d_words)
        seq2 = get_vector(q_words)
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
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
    new_in_x3 = []
    for i in in_x3:
        #print i
        new_in_x3 += i
    #print new_in_x3
    return in_x1, in_x2, new_in_x3, in_y


def vectorize_all(examples, word_dict, char_dict, ner_dict,
              sort_by_len=True, verbose=True, concat=False):
    """
        examples:documents, questions, options, answers, doc_char, ques_char, opt_char, 
                            doc_ners, ques_ners, opt_ners, document_co_occur_features
        Vectorize all features including word embedding, char embedding and ner embedding
        in_x1: word embeddings for doc
        in_x2: word embeddings for ques
        in_x3: word embeddings for options
        in_x4: char embeddings for doc
        in_x5: char embeddings for ques
        in_x6: char embeddings for options
        in_x7: ner embeddings for doc
        in_x8: ner embeddings for ques
        in_x9: ner embeddings for options
        in_x10: co-features for doc
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_x3 = []
    in_x4 = []
    in_x5 = []
    in_x6 = []
    in_x7 = []
    in_x8 = []
    in_x9 = []
    in_x10 = []
    in_y = []
    def get_vector(st, dic):
        seq = [dic[w] if w in dic else 1 for w in st]
        return seq

    def get_vector_char(st, dic):
        seq = [[dic[c] if c in dic else 1 for c in w] for w in st]
        return seq
    op_w = examples[2]
    op_c = examples[6]
    op_ner = examples[9]
    examples = list(examples[0:2]) + [examples[3]] + list(examples[4:6]) + list(examples[7:9]) + [examples[10]]
    for idx, (d, q, a, d_c, q_c, d_ner, q_ner, d_co_f) in enumerate(zip(*examples)):
        d_words = d.split(' ')
        q_words = q.split(' ')
        d_char = d_c.split(' ')
        q_char = q_c.split(' ')
        d_ner = d_ner.split(' ')
        q_ner = q_ner.split(' ')
        # assert 0 <= a <= 3
        seq1 = get_vector(d_words, word_dict)
        seq2 = get_vector(q_words, word_dict)
        seq3 = get_vector_char(d_char, char_dict)
        seq4 = get_vector_char(q_char, char_dict)
        seq5 = get_vector(d_ner, ner_dict)
        seq6 = get_vector(q_ner, ner_dict)
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1 += [seq1]
            in_x2 += [seq2]
            in_x4 += [seq3]
            in_x5 += [seq4]
            in_x7 += [seq5]
            in_x8 += [seq6]
            in_x10 += [d_co_f]
            option_seq = []
            option_char_seq = []
            option_ner_seq = []
            for i in range(4):
                if concat:
                    op = " ".join(q_words) + ' @ ' + op_w[i + idx * 4]
                    op_char_one = q_c + ' @ ' + op_c[i + idx * 4]
                    op_ner_one = " ".join(q_ner) + ' @ ' + op_ner[i + idx * 4]
                else:
                    op = op_w[i + idx * 4]
                    op_char_one = op_c[i + idx * 4]
                    op_ner_one = op_ner[i + idx * 4]
                op = op.split(' ')
                option = get_vector(op, word_dict)
                op_char_one = op_char_one.split(' ')
                op_char_one = get_vector_char(op_char_one, char_dict)
                op_ner_one = op_ner_one.split(' ')
                op_ner_one = get_vector(op_ner_one, ner_dict)
                assert len(option) > 0
                option_seq += [option]
                option_char_seq += [op_char_one]
                option_ner_seq += [op_ner_one]

            in_x3 += [option_seq]
            in_x6 += [option_char_seq]
            in_x9 += [option_ner_seq]
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
        in_x3 = [in_x3[i] for i in sorted_index]
        in_x4 = [in_x4[i] for i in sorted_index]
        in_x5 = [in_x5[i] for i in sorted_index]
        in_x6 = [in_x6[i] for i in sorted_index]
        in_x7 = [in_x7[i] for i in sorted_index]
        in_x8 = [in_x8[i] for i in sorted_index]
        in_x9 = [in_x9[i] for i in sorted_index]
        in_x10 = [in_x10[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
    new_in_x3 = []
    new_in_x6 = []
    new_in_x9 = []
    for i in in_x3:
        new_in_x3 += i
    for i in in_x6:
        new_in_x6 += i
    for i in in_x9:
        new_in_x9 += i
    return in_x1, in_x2, new_in_x3, in_x4, in_x5, new_in_x6, in_x7, in_x8, new_in_x9, in_x10, in_y


def prepare_data(seqs, padding=None, dynamic=False, mask=True):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    if padding is None:
        max_len = np.max(lengths)
    else:
        max_len = padding
        if dynamic:
            max_len = min(np.max(lengths), padding)
    x = np.zeros((n_samples, max_len)).astype(np.int32)
    if mask:
        x_mask = np.zeros((n_samples, max_len)).astype(np.float32)
        for idx, seq in enumerate(seqs):
            x[idx, :min(max_len, lengths[idx])] = seq[:min(max_len, lengths[idx])]
            x_mask[idx, :min(max_len, lengths[idx])] = 1.0
        return x, x_mask
    else:
        for idx, seq in enumerate(seqs):
            x[idx, :min(max_len, lengths[idx])] = seq[:min(max_len, lengths[idx])]
        return x


def prepare_data_char(seqs, args, padding=None, dynamic=False, mask=True):
    lengths = [len(seq) for seq in seqs]
    word_lengths = [[len(w) for w in seq] for seq in seqs]
    n_samples = len(seqs)
    max_char_len = args.max_word_len
    if padding is None:
        max_len = np.max(lengths)
    else:
        max_len = padding
        if dynamic:
            max_len = min(np.max(lengths), padding)
    x = np.zeros((n_samples, max_len, max_char_len)).astype(np.int32)
    if mask:
        x_mask = np.zeros((n_samples, max_len, max_char_len)).astype(np.float32)
        for idx, seq in enumerate(seqs):
            for jdx, word in enumerate(seq):
                if jdx >= max_len:
                    break
                x[idx, jdx, :min(max_char_len, word_lengths[idx][jdx])] = seq[jdx][:min(max_char_len, word_lengths[idx][jdx])]
            x_mask[idx, jdx, :min(max_char_len, word_lengths[idx][jdx])] = 1.0
        return x, x_mask
    else:
        for idx, seq in enumerate(seqs):
            for jdx, word in enumerate(seq):
                if jdx >= max_len:
                    break
                x[idx, jdx, :min(max_char_len, word_lengths[idx][jdx])] = seq[jdx][:min(max_char_len, word_lengths[idx][jdx])]
        return x


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
        embeddings[0] = np.zeros((dim, ))
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
