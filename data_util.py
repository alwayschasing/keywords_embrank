#!/usr/bin/env python
# -*- coding:utf-8 -*-
import collections

DocItem = collections.namedtuple("DocItem", ["id","title","content"])
def load_articles(input_file):
    articles = [] 
    fp = open(input_file, "r", encoding="utf-8")
    for line in fp:
        parts = line.strip().split('\t')
        docid = parts[0]
        title = parts[1]
        content = parts[2]
        item = DocItem(docid, title, content)
        articles.append(item)
    return articles