#!/usr/bin/env python
# -*- coding:utf-8 -*-

def load_vocab(vocab_file):
    vocab = set()
    fp = open(vocab_file, "r", encoding="utf-8")
    for line in fp:
        word = line.rstrip('\n').split('\t')[0]
        vocab.add(word)
    return vocab

def statistic_word_freq(vocab_file, cut_corpus_file, output_file):
    vocab = load_vocab(vocab_file)
    rfp = open(cut_corpus_file, "r", encoding="utf-8")

    word_count = {}
    for line in rfp:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        title = parts[1]
        content = parts[2]
        
        text = title + ' ' + content 
        cut_text = text.split(' ')
        for w in cut_text:
            if w in vocab:
                if w in word_count:
                    word_count[w] += 1
                else:
                    word_count[w] = 1

    wfp = open(output_file, "w", encoding="utf-8")
    for k,v in word_count.items():
        wfp.write("%s\t%d\n" % (k,v))
    wfp.close()
    rfp.close()

if __name__ == "__main__":
    vocab_file = "../config_data/articles_7d_vocab.txt"
    cut_corpus_file = "preprocess_res/cut_articles_7d.tsv"
    output_file = "preprocess_res/word_freq.txt"
    statistic_word_freq(vocab_file, cut_corpus_file, output_file)

     