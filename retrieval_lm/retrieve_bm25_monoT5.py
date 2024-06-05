import sys
import os

from typing import Any
from pyserini.search.lucene import LuceneSearcher
import torch
import transformers
import numpy as np

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(ROOT_PATH)

from monoT5 import MonoT5


class Retriever:
    def __init__(self, index="miracl-v1.0-en"):
        self.docs_ids = []
        self.searcher = LuceneSearcher.from_prebuilt_index(index)
        self.ranker = MonoT5(device="cuda")

    def search(self, query, k=1):
        docs = self.searcher.search(query, k=100)
        retrieved_docid = [i.docid for i in docs]
        docs_text = [
            eval(self.searcher.doc(docid).raw())
            for j, docid in enumerate(retrieved_docid)
        ]
        ranked_doc = self.ranker.rerank(query, docs_text)[:20]
        docids = [i["docid"] for i in ranked_doc]
        docs_text = [self.searcher.doc(docid).raw() for j, docid in enumerate(docids)]
        return docids, docs_text

    def process(self, query, **kwargs):
        docs_text = self.search(query, **kwargs)
        return f"\n[DOCS] {docs_text} [/DOCS]\n"
