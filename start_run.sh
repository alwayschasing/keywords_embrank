#!/bin/bash

python extract_keywords.py \
    --vocab="config_data/articles_7d_vocab.txt" \
    --vocab_emb="config_data/articles_7d_word2vec.txt" \
    --keywords_vocab="keyword_vocab_final" \
    --input_file="/search/odin/liruihong/keyword-project/input_data/test_articles.tsv" \
    --output_file="output_data/test_keywords.txt"