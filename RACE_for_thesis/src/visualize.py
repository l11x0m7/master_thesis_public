# -*- encoding:utf-8 -*-
import os
import numpy as np
import cPickle as pkl
import json

def pick_article(words):
    difficulty_set = ["high"]
    raw_data = "../data/RACE"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    avg_article_sentence_count = 0
    max_article_sentence_count = -1
    min_article_sentence_count = 999
    num_que = 0
    for data_set in ["test"]:
        for d in difficulty_set:
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                article_words = set(obj['article'].lower().split())
                w_c = 0
                for w in words:
                    if w not in article_words:
                        break
                    w_c += 1
                if w_c == len(words):
                    print obj['article']
                    print obj['questions']
                    print obj['options']
                    print



def find_match():
    sentence_store, sentence_att, \
    sentence_article, sentence_question, \
    sentence_options, sentence_answers, prediction = pkl.load(open('cache/visualization.pkl'))

    count = 0
    thres = 0.2
    for sent_article, q, o, a, \
            sent_att, res_att, pred in \
                zip(sentence_article, sentence_question,
                sentence_options, sentence_answers, \
                sentence_store, sentence_att, prediction):
        if pred == a == np.argmax(sent_att):
            if np.sum(np.asarray(res_att) >= thres) >= 2 and np.sum(np.asarray(res_att) > 0) > 8:
                sentences = [' '.join(sent).replace('<PAD>', '').strip(' ') for sent in sent_article]
                for i, (sent, att) in enumerate(zip(sentences, res_att)):
                    if att >= thres:
                        sentences[i] = '\033[1;31;40m{}\033[0m'.format(sent)
                print '|||'.join(sentences)
                print ' '.join(q).replace('<PAD>', '').strip()
                print ' '.join(o[0]).replace('<PAD>', '').strip()
                print ' '.join(o[1]).replace('<PAD>', '').strip()
                print ' '.join(o[2]).replace('<PAD>', '').strip()
                print ' '.join(o[3]).replace('<PAD>', '').strip()
                print a
                print sent_att
                print res_att
                print
                count += 1
                # if count == 10:
                #     break
    print count


pick_article(['heavy', 'suitcases'])
# find_match()
