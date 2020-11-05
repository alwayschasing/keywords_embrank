#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname).1s %(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d_%H:%M:%S")
logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    vocab = dict()
    fp = open(vocab_file, "r", encoding="utf-8")
    for idx, line in enumerate(fp):
        word = line.rstrip('\n').split('\t')[0]
        vocab[word] = idx
    return vocab

def load_word_freq(vocab_file):
    vocab = dict()
    fp = open(vocab_file, "r", encoding="utf-8")
    for idx, line in enumerate(fp):
        word,freq = line.rstrip('\n').split('\t')
        vocab[word] = int(freq)
    return vocab

def generate_kw_vocab(word_freq_file, ori_vocab_file, output_vocab_file):
    ori_vocab = load_vocab(ori_vocab_file)
    word_freq = load_word_freq(word_freq_file)

    new_vocab = []
    for word in ori_vocab.keys():
        if word not in word_freq:
            continue

        if word_freq[word] < 150:
            continue
        new_vocab.append(word)

    wfp = open(output_vocab_file, "w", encoding="utf-8")
    for word in new_vocab:
        wfp.write("%s\n" % (word))
    wfp.close()

if __name__ == "__main__":
    word_freq_file = "preprocess_res/word_freq.txt"
    ori_vocab_file = "../config_data/keyword_vocab_final"
    output_vocab_file = "preprocess_res/new_vocab.txt"
    generate_kw_vocab(word_freq_file, ori_vocab_file, output_vocab_file)
