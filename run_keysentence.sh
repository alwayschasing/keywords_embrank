#!/bin/bash

vocab_freq="config_data/word_freq.txt"
python extract_keywords.py \
    --debug=1 \
    --vocab="config_data/articles_7d_vocab.txt" \
    --vocab_emb="config_data/articles_7d_word2vec.txt" \
    --keywords_vocab="config_data/keyword_vocab_final" \
    --vocab_freq=$vocab_freq \
    --input_file="/search/odin/liruihong/keyword-project/input_data/test_articles.tsv" \
    --output_file="output_data/test_sif_keywords.txt"
