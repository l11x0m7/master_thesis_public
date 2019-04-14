import os
import json
import re
import time
import numpy as np
from collections import Counter
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP

stopwords = stopwords.words('english')
english_punctuations = [',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%']
stopwords += english_punctuations


nlp = StanfordCoreNLP(r'/home/linxuming/nltk_data/stanford/stanford-corenlp-full-2016-10-31/', memory='8g')

import sys
reload(sys)
sys.setdefaultencoding('utf8')

# wordnet.synsets(word) shows the synonym words

def stanford_tokenize(st, lower=True):
    ans = []
    for sent in sent_tokenize(st):
        ans += nlp.word_tokenize(sent)
    if lower:
        return " ".join(ans).lower().strip()
    else:
        return " ".join(ans).strip()

def stanford_postag(st):
    ans = []
    for sent in sent_tokenize(st):
        word, postag = zip(*nlp.pos_tag(sent))
        ans += postag
    return " ".join(ans)

def stanford_ner(st):
    ans = []
    for sent in sent_tokenize(st):
        word, ner = zip(*nlp.ner(sent))
        ans += ner
    return " ".join(ans)

def stanford_sentence_tokenize(st):
    def parse(r_dict):
        words = []
        tags = []
        ners = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['word'])
        return ' '.join(words)
    props={'annotators': 'ssplit,tokenize','pipelineLanguage':'en','outputFormat':'json'}

    ans = []
    for sent in st.strip().split('\n'):
        ans += sent_tokenize(sent)
    for i, sent in enumerate(ans):
        tokenized_words = nlp.annotate(sent, properties=props)
        tokenized_words = json.loads(tokenized_words, strict=False)
        ans[i] = parse(tokenized_words).lower().strip()
    return '|||'.join(ans)

def sentence_tokenize(st):
    ans = []
    for sent in st.strip().split('\n'):
        ans += sent_tokenize(sent)
    for i, sent in enumerate(ans):
        ans[i] = ' '.join(word_tokenize(sent)).lower().strip()
    return '|||'.join(ans)

def tokenize(st, lower=True):
    #TODO: The tokenizer's performance is suboptimal
    ans = []
    for sent in sent_tokenize(st):
        ans += word_tokenize(sent)
    if lower:
        return " ".join(ans).lower().strip()
    else:
        return " ".join(ans).strip()

def get_key_sentence(sentences, questions, options, method=0):
    """
    method: 0 for the same or similar words, 1 for the NER words
    """
    sentences = sentences.split('|||')
    wl = WordNetLemmatizer()
    key_sentences_per_question = []
    for i, q in enumerate(questions):
        filters = []
        filters.append(q)
        for o in options[i]:
            filters.append(o)
        filter_set = set()
        if method == 0:
            for sentence in filters:
                words_and_postags = nlp.pos_tag(sentence)
                for word, postag in words_and_postags:
                    if not postag.startswith('NN') \
                    and not postag.startswith('VB') \
                    and postag != 'ADJ' \
                    and postag != 'ADV' \
                    and postag != 'MOD':
                        continue
                    # print word, postag
                    lemma_word = wl.lemmatize(word)
                    if lemma_word not in stopwords and lemma_word != '':
                        filter_set.add(lemma_word)
        else:
            for sentence in filters:
                word_ner = nlp.ner(sentence)
                # print(word_ner)
                for word, ner in word_ner:
                    lemma_word = wl.lemmatize(word)
                    if ner != 'O' and lemma_word not in stopwords and lemma_word != '':
                        filter_set.add(lemma_word)

        filter_sentences = set()
        for i, sentence in enumerate(sentences):
            if len(set([wl.lemmatize(word) for word in sentence.split()]) & filter_set) != 0:
                # print(set([wl.lemmatize(word) for word in sentence.split()]) & filter_set)
                filter_sentences.add(i)
                if i > 0:
                    filter_sentences.add(i-1)
                if i < len(sentences) - 1:
                    filter_sentences.add(i+1)
        filter_sentences = list(sorted(list(filter_sentences)))
        if len(filter_sentences) == 0:
            filter_sentences = range(len(sentences))
        key_sentences_per_question.append(filter_sentences)
    return key_sentences_per_question

def get_cooccur_feature(article, questions, options):
    """
    Get the cooccurance feature for the article. 
    Different questions with the same article can get different results.
    """
    article = article.split()
    wl = WordNetLemmatizer()
    article_co_feature_per_question = []
    for i, q in enumerate(questions):
        filters = []
        filters.append(q)
        for o in options[i]:
            filters.append(o)
        filter_set = set()
        for sentence in filters:
            for word, postag in nlp.pos_tag(sentence):
                if not postag.startswith('NN') \
                and not postag.startswith('VB') \
                and postag != 'ADJ' \
                and postag != 'ADV' \
                and postag != 'MOD':
                    continue
                # print word, postag
                lemma_word = wl.lemmatize(word)
                if lemma_word not in stopwords and lemma_word != '':
                    filter_set.add(lemma_word)

        article_co_feature = []
        for word in article:
            if wl.lemmatize(word) in filter_set:
                article_co_feature.append(1)
            else:
                article_co_feature.append(0)
        article_co_feature_per_question.append(article_co_feature)

    return article_co_feature_per_question


def get_position_feature(article, questions, options, window_span=20):
    """
    Get the cooccurance feature for the article. 
    Different questions with the same article can get different results.
    """
    article = article.split()
    wl = WordNetLemmatizer()
    article = [wl.lemmatize(_) for _ in article]
    article_position_feature_per_question = []
    for i, q in enumerate(questions):
        filters = []
        filters.append(q)
        for o in options[i]:
            filters.append(o)
        filter_set = set()
        for sentence in filters:
            for word, postag in nlp.pos_tag(sentence):
                if not postag.startswith('NN') \
                and not postag.startswith('VB') \
                and postag != 'ADJ' \
                and postag != 'ADV' \
                and postag != 'MOD':
                    continue
                # print word, postag
                lemma_word = wl.lemmatize(word)
                if lemma_word not in stopwords and lemma_word != '':
                    filter_set.add(lemma_word)

        article_co_feature = []
        for pos, word in enumerate(article):
            pos_feature = [0] * window_span
            for dis in xrange(window_span):
                if dis == 0:
                    if word in filter_set:
                        pos_feature[dis] = 1
                elif pos - dis >= 0 and article[pos - dis] in filter_set:
                    pos_feature[dis] += 1
                elif pos + dis < len(article) and article[pos - dis] in filter_set:
                    pos_feature[dis] += 1
            article_position_feature.append(pos_feature)
        article_position_feature_per_question.append(article_position_feature)

    return article_position_feature_per_question


def get_ner_feature(sentence):
    try:
        _, ners = zip(*nlp.ner(sentence))
    except:
        ners = ['O' for _ in xrange(len(sentence))]
    return ners 


def get_postag_feature(sentence):
    try:
        _, postags = zip(*nlp.pos_tag(sentence))
    except:
        print(sentence)
        print('Error from get_postag_feature')
        exit(1)
    return postags


def get_all_nlp_feature(sentence):
    def parse(r_dict):
        words = []
        tags = []
        ners = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['word'])
                tags.append(token['pos'])
                ners.append(token['ner'])
        return ' '.join(words), ' '.join(tags), ' '.join(ners)
    props={'annotators': 'tokenize,pos,ner','pipelineLanguage':'en','outputFormat':'json'}
    r_text = nlp.annotate(sentence.strip(), properties=props)
    r_dict = json.loads(r_text, strict=False)
    parse_res = parse((r_dict))
    return parse_res



def main():
    difficulty_set = ["middle", "high"]
    data = "../data/my_data"
    if not os.path.exists(data):
        os.mkdir(data)
    raw_data = "../data/RACE"
    cnt = 0
    avg_article_length = 0
    avg_question_length = 0
    avg_option_length = 0
    avg_article_sentence_count = 0
    max_article_sentence_count = -1
    min_article_sentence_count = 999
    num_que = 0
    tokenizer = stanford_tokenize
    for data_set in ["train", "dev", "test"]:
        p1 = os.path.join(data, data_set)
        if not os.path.exists(p1):
            os.mkdir(p1)
        for d in difficulty_set:
            print("Starting on {}/{}".format(data_set, d))
            new_data_path = os.path.join(data, data_set, d)
            if not os.path.exists(new_data_path):
                os.mkdir(new_data_path)
            new_raw_data_path = os.path.join(raw_data, data_set, d)
            for inf in os.listdir(new_raw_data_path):
                cnt += 1
                obj = json.load(open(os.path.join(new_raw_data_path, inf), "r"))
                obj["article"] = obj["article"].replace("\\newline", "\n")
                obj["article"] = re.sub(r"\n+", r"\n", obj["article"])
                # obj['article'] = obj['article'].replace('\n', ' ')
                # obj['article'] = re.sub(r' [2,]', r'. ', obj['article'])
                # obj['article'] = obj['article'].replace('.', '. ')
                obj['sent_article'] = stanford_sentence_tokenize(obj['article'])
                # obj['article_char'] = tokenizer(obj['article'], False)
                # obj['article_ner'] = get_ner_feature(obj['article_char'])
                # obj['article_postag'] = get_postag_feature(obj['article_char'])
                obj['article_char'], obj['article_postag'], obj['article_ner'] = \
                                                    get_all_nlp_feature(obj['article'])
                obj["article"] = obj['article_char'].lower()
                # print(len(obj['article_char'].split()), len(obj['article_ner']), 
                    # len(obj['article_postag']))
                avg_article_length += obj["article"].count(" ")
                avg_article_sentence_count += (obj['sent_article'].count("|||") + 1)
                min_article_sentence_count = min(min_article_sentence_count, obj['sent_article'].count('|||') + 1)
                max_article_sentence_count = max(max_article_sentence_count, obj['sent_article'].count('|||') + 1)
                obj['questions_ner'] = []
                obj['questions_postag'] = []
                obj['questions_char'] = []
                obj['options_ner'] = []
                obj['options_postag'] = []
                obj['options_char'] = []
                for i in range(len(obj["questions"])):
                    num_que += 1
                    # obj["questions_char"].append(tokenizer(obj["questions"][i], False))
                    # obj['questions_ner'].append(get_ner_feature(obj['questions_char'][i]))
                    # obj['questions_postag'].append(get_postag_feature(obj['questions_char'][i]))
                    obj["questions_char"].append([])
                    obj['questions_ner'].append([])
                    obj['questions_postag'].append([])
                    obj['questions_char'][i], obj['questions_postag'][i], obj['questions_ner'][i] = \
                                                    get_all_nlp_feature(obj['questions'][i])
                    obj["questions"][i] = obj['questions_char'][i].lower()
                    avg_question_length += obj["questions"][i].count(" ")
                    # print(len(obj['questions_char'][i].split()), len(obj['questions_ner'][i]), 
                        # len(obj['questions_postag'][i]))
                    obj['options_ner'].append([])
                    obj['options_postag'].append([])
                    obj['options_char'].append([])
                    for k in range(4):
                        obj['options_ner'][i].append([])
                        obj['options_postag'][i].append([])
                        obj['options_char'][i].append([])
                        obj['options_char'][i][k], obj['options_postag'][i][k], obj['options_ner'][i][k] = \
                                                    get_all_nlp_feature(obj['options'][i][k])
                        # obj["options_char"][i].append(tokenizer(obj["options"][i][k], False))
                        # obj['options_ner'][i].append(get_ner_feature(obj['options_char'][i][k]))
                        # obj['options_postag'][i].append(get_postag_feature(obj['options_char'][i][k]))
                        obj["options"][i][k] = obj["options_char"][i][k].lower()
                        avg_option_length += obj["options"][i][k].count(" ")
                        # print(len(obj['options_char'][i][k].split()), len(obj['options_ner'][i][k]), 
                            # len(obj['options_postag'][i][k]))
                obj['key_sent'] = get_key_sentence(obj['sent_article'], obj['questions'], obj['options'])
                obj['article_cooccur'] = get_cooccur_feature(obj['article'], obj['questions'], obj['options'])
                json.dump(obj, open(os.path.join(new_data_path, inf), "w"), indent=4)
                if cnt % 100 == 0:
                    print('Finished {}'.format(cnt))
    print "avg article length", avg_article_length * 1. / cnt
    print "avg question length", avg_question_length * 1. / num_que
    print "avg option length", avg_option_length * 1. / (num_que * 4)
    print "avg article sentence number", avg_article_sentence_count * 1. / cnt
    print "max article sentence number", max_article_sentence_count
    print "min article sentence number", min_article_sentence_count


def eval_data():
    in_file_train = '../data/my_data/train'
    in_file_dev = '../data/my_data/dev'
    in_file_test = '../data/my_data/test'
    documents = []
    sent_documents = []
    questions = []
    answers = []
    options = []
    doc_len = []
    question_len = []
    option_len = []
    num_examples = 0
    key_sent_lengths = []
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
    files = get_file(in_file_train)
    files += get_file(in_file_dev)
    files += get_file(in_file_test)
    for inf in files:
        obj = json.load(open(inf, "r"))
        # print obj['sent_article'], obj['questions'], obj['options']
        # return
        doc_len.append(len(obj['article'].split()))
        for i, q in enumerate(obj["questions"]):
            assert len(obj['article_cooccur'][i]) == doc_len[-1]
            documents += [obj["article"]]
            sent_documents += [obj['sent_article']]
            questions += [q]
            assert len(obj["options"][i]) == 4
            options += obj["options"][i]
            answers += [ord(obj["answers"][i]) - ord('A')]
            key_sent_lengths.append(len(obj['key_sent'][i]))
            question_len.append(len(q.split()))
            for _ in obj['options'][i]:
                option_len.append(len(_.split()))
            # if len(obj['key_sent'][i]) == 0:
                # print(q)
            num_examples += 1
    def clean(st_list):
        for i, st in enumerate(st_list):
            st_list[i] = st.lower().strip()
        return st_list
    documents = clean(documents)
    sent_documents = clean(sent_documents)
    questions = clean(questions)
    options = clean(options)
    sent_lengths = [len(sample.split('|||')) for sample in sent_documents]
    word_lengths = [len(sent.split(' ')) for sample in sent_documents for sent in sample.split('|||')]
    print(sent_lengths[:10])
    print(word_lengths[:10])
    print(max(sent_lengths))
    print(max(word_lengths))
    print(np.percentile(sent_lengths, 90))
    print(np.percentile(word_lengths, 90))
    print(np.percentile(key_sent_lengths, 90))
    print(np.percentile(sent_lengths, 95))
    print(np.percentile(word_lengths, 95))
    print(np.percentile(key_sent_lengths, 95))
    print(np.percentile(sent_lengths, 98))
    print(np.percentile(word_lengths, 98))
    print(np.percentile(key_sent_lengths, 98))
    print(np.mean(sent_lengths))
    print(np.median(sent_lengths))
    print(np.max(sent_lengths))
    print(np.mean(key_sent_lengths))
    print(np.median(key_sent_lengths))
    print(np.max(key_sent_lengths))
    print(len([i for i in key_sent_lengths if i == 0]))

    print('statistics for doc, question and option lengths')
    print(np.mean(doc_len))
    print(np.max(doc_len))
    print(np.min(doc_len))
    for _ in range(90, 100):
        print(np.percentile(doc_len, _))
    print(np.mean(question_len))
    print(np.max(question_len))
    print(np.min(question_len))
    print(np.percentile(question_len, 98))
    print(np.mean(option_len))
    print(np.max(option_len))
    print(np.min(option_len))
    print(np.percentile(option_len, 98))
    return (documents, questions, options, sent_documents, answers)

if __name__ == "__main__":
    t1 = time.time()
    eval_data()
    # main()
    print('Time consuming:', time.time() - t1)
    # sentences = "today , an increasing number of people are always looking at their mobile phones with their heads down .|||these people are called the `` heads-down tribe '' .|||are you a heads-down tribe member ?|||heads-down tribe members now can be seen everywhere .|||using mobile phones may cause accidents and even cost a lot of money .|||also , more and more interesting and strange facts happen to the `` heads-down tribe '' .|||let 's have a look at an interesting tv report .|||a man in america kept using his mobile phone on his way home .|||as a result , he bumped into a big lost bear on the street .|||when he lifted his eyes from the phone , he was so scared that he turned around and ran away as quickly as possible .|||another fact is that we can often see people in the restaurant eating face to face but looking at their own mobile phones .|||it 's strange that they do n't talk to the ones who sit opposite to them during the meal .|||some of them even have fun communicating with others on the phone .|||mobile phones are helpful and necessary tools for modern life .|||are mobile phones good or bad ?|||it depends on how people use them .|||let 's be `` healthy '' users and try to be the `` heads-up tribe '' . "
    # questions = [u'where did the american bump into the bear ?', u"according to the passage , what do the `` heads-down tribe '' do when they eat in a restaurant ?", u"what do we know about the `` heads-down tribe '' ?", u'what is the main idea of the passage ?'] 
    # options = [[u'in the forest .', u'at a zoo .', u'on the street .', u'at a park .'], [u'they enjoy their meals .', u'they take photos of their meals .', u"they do n't talk to the people who sit opposite .", u'they talk to the strangers who sit next to them .'], [u'they are cool children .', u'they are dangerous driver .', u'they are good at using the internet .', u'they are always looking at their mobile phones .'], [u"let 's try to be the `` heads-down tribe '' .", u'we should use mobile phones properly .', u'mobile phones are not good for people .', u'mobile phones are helpful and necessary .']]
    # print get_key_sentence(sentences, questions, options, method=0)
