from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import jieba.posseg as pseg
import codecs
import os
import thulac
from . import util

logger = util.set_logger()
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
#sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']

def get_default_stop_words_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, 'stopwords.txt')

class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class WordSegmentation(object):
    """ 分词 """
    def __init__(self, stop_words_file=None, allow_speech_tags=allow_speech_tags, user_dict=None, vocab=None, use_speech_tag=False):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """
        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        self.stop_words_file = get_default_stop_words_file()
        self.vocab=vocab
        if stop_words_file is not None:
            for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
                self.stop_words.add(word.strip())
        model_path="/search/odin/liruihong/keyword-project/SIFRank_zh/auxiliary_data/thulac.models"
        if use_speech_tag == False:
            self.segmenter = thulac.thulac(model_path=model_path, user_dict=user_dict, seg_only=True)
        else:
            self.segmenter = thulac.thulac(model_path=model_path, user_dict=user_dict)
        self.use_speech_tag = use_speech_tag


    def segment(self, text, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """对一段文本进行分词，返回list类型的分词结果

        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。
        """
        #jieba_result = pseg.cut(text)
        cut_result = self.segmenter.cut(text)

        use_speech_tags_filter = self.use_speech_tag
        if use_speech_tags_filter == True:
            cut_result = [w for w in cut_result if w[1] in self.default_speech_tag_filter]
        else:
            cut_result = [w for w in cut_result]

        # 去除特殊符号
        word_list = [w[0].strip() for w in cut_result if w[1] != 'x']
        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        logger.debug("[check segment] %s" % (word_list))
        return word_list

    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。
        sequences -- 列表，每个元素是一个句子（字符串类型）
        """
        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence,
                                    lower=lower,
                                    use_stop_words=use_stop_words,
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation(object):
    """ 分句 """
    def __init__(self, delimiters=sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set(delimiters)

    def segment(self, text):
        res = [text]
        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res


class Segmentation(object):
    def __init__(self, user_dict=None,
                 stop_words_file=None,
                 allow_speech_tags=allow_speech_tags,
                 delimiters=sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags, user_dict=user_dict)
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences,
                                                    lower=lower,
                                                    use_stop_words=False,
                                                    use_speech_tags_filter=False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences,
                                                        lower=lower,
                                                        use_stop_words=True,
                                                        use_speech_tags_filter=False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences,
                                                      lower=lower,
                                                      use_stop_words=True,
                                                      use_speech_tags_filter=True)

        return AttrDict(
            sentences=sentences,
            words_no_filter=words_no_filter,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )


if __name__ == '__main__':
    pass
