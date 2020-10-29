#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import os
os.environ["DEBUG"] = "1"
import argparse
from model import KeywordsBySenTextrank
import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0, help="whether to debug")
parser.add_argument('--vocab', type=str, default=None, help="vocab file")
parser.add_argument('--vocab_emb', type=str, default=None, help="vocab emb file")
parser.add_argument('--keywords_vocab', type=str, default=None, help="keywords vocab file")

parser.add_argument('--input_file', type=str, default=None, required=True, help="input file")
parser.add_argument('--output_file', type=str, default=None, required=True, help="output file")
args = parser.parse_args()

#### logger set
loglevel=logging.INFO
if os.environ["DEBUG"] == "1":
    loglevel=logging.DEBUG
logging.basicConfig(level=loglevel, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")
logger = logging.getLogger(__name__)
####


def main(args):
    extractor = KeywordsBySenTextrank(vocab_file=args.vocab,
                                      vocab_emb_file=args.vocab_emb,
                                      keywords_file=args.keywords_vocab,
                                      user_dict=args.keywords_vocab)

    articles = data_util.load_articles(args.input_file)
    articles = articles[0:100]
    wfp = open(args.output_file, "w", encoding="utf-8")
    for doc_item in articles:
        doc_id = doc_item.id
        doc_title = doc_item.title
        doc_content = doc_item.content

        doc_text = doc_title + "。" + doc_content
        keyword_score = extractor.get_keyword_score(doc_text)
        debug_line = "\t".join(["%s:%f" % (kw,score) for kw,score in keyword_score])
        logger.debug("%s\n[keywords]%s" % (doc_text, debug_line))

        write_line = " ".join(["%s:%f" % (kw,score) for kw,score in keyword_score])
        wfp.write("%s\t%s\n" % (doc_id, write_line))
    wfp.close()


if __name__ == "__main__":
    main(args)