from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import networkx as nx
import scipy
import gensim
import numpy as np
import os
import logging
from . import util
from .segmentation import Segmentation
loglevel=logging.INFO
if os.environ["DEBUG"] == 1:
    loglevel=logging.DEBUG
logging.basicConfig(level=loglevel, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")
logger = logging.getLogger(__name__)


class KeywordsBySenTextrank(object):
    def __init__(self,
                 vocab_file,
                 vocab_emb_file,
                 keywords_file=None,
                 user_dict=None,
                 stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，停止词文件路径，若不是str则是使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        Object Var:
        self.words_no_filter         --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words     --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters       --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.seg = Segmentation(user_dict=user_dict,
                                stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)

        self.vocab = util.load_vocab(vocab_file)
        logger.info("load vocab:%s, vocab size:%d" % (vocab_file, len(self.vocab)))
        if keywords_file is None:
            self.keywords_vocab = None
            logger.info("keywords_vocab is None")
        else:
            self.keywords_vocab = util.load_vocab(keywords_file)
            logger.info("load keywords vocab:%s, keywords_vocab size:%d" %(keywords_file, len(self.keywords_vocab)))

        #self.emb_vocab = gensim.models.KeyedVectors.load_word2vec_format(vocab_emb_file)
        self.emb_vocab = util.load_word2vec_emb(vocab_emb_file)
        assert len(self.emb_vocab) == len(self.vocab)
        logger.info("load vocab emb %s, emb vec size:%d" % (vocab_emb_file, len(self.emb_vocab.items()[0][1])))

    def get_sentence_textrank(self, sentences, words, sim_func, pagerank_config={'alpha': 0.85}):
        """将句子按照关键程度从大到小排序
            Keyword arguments:
            sentences         --  列表，元素是句子
            words             --  二维列表，子列表和sentences中的句子对应，子列表由单词组成
            sim_func          --  计算两个句子的相似性，参数是两个由单词组成的列表
            pagerank_config   --  pagerank的设置
        """
        _source = words
        sentences_num = len(sentences)
        graph = np.zeros((sentences_num, sentences_num))

        for x in range(sentences_num):
            for y in range(x, sentences_num):
                similarity = sim_func(sentences[x], sentences[y])
                graph[x, y] = similarity
                graph[y, x] = similarity

        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **pagerank_config)  # this is a dict

        sentence_score = []
        for index, score in scores.items():
            sentence_score.append(util.Item(index=index, score=score))
        return sentence_score

    def get_keyword_score(self, text, lower=True, source='no_stop_words'):
        result = self.seg.segment(text=text, lower=lower)
        sentences = result.sentences
        #words_no_filter = result.words_no_filter
        #words_no_stop_words = result.words_no_stop_words
        #words_all_filters = result.words_all_filters
        options = ['no_filter', 'no_stop_words', 'all_filters']
        if source in options:
            _source = result['words_' + source]
        else:
            _source = result['words_no_stop_words']

        sentence_vecs = []
        for sen in sentences:
            word_vecs = []
            for w in sen:
                word_vecs.append(self.emb_vocab[w])
            word_vecs = np.asarray(word_vecs)
            sen_vec = np.mean(word_vecs, asix=-1)
            sentence_vecs.append(sen_vec)

        # list of dict item, item.index, item.score
        sentence_score = self.get_sentence_textrank(sentences=sentence_vecs,
                                                    sim_func=util.cosine_similar,
                                                    pagerank_config={'alpha': 0.85})
        word_score = {}
        for sen_item in sentence_score:
            sen_idx = sen_item.index
            sen_score = sen_item.socre

            sen_words = sentences[sen_idx]
            for w in sen_words:
                if w in word_score:
                    if word_score[w] < sen_score:
                        tmp_score = 1 - scipy.spatial.distance.cosine(self.emb_vocab[w], sentence_vecs[sen_idx])*sen_score
                        word_score[w] = tmp_score
                else:
                    tmp_score = 1 - scipy.spatial.distance.cosine(self.emb_vocab[w], sentence_vecs[sen_idx])*sen_score
                    word_score[w] = tmp_score

        sort_word_score = sorted(word_score.items(), lambda x:x[1], reverse=True)
        return sort_word_score


if __name__ == '__main__':
    pass
