#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import listdir, getcwd, walk, mkdir, chdir, environ
from os.path import join, isdir, realpath, relpath, exists, getsize
from json import load, dump, dumps
from statistics import mean
from math import inf, log, sqrt
from sys import stdout, getsizeof
from re import match, split, findall, sub
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from time import time, sleep
from datetime import datetime
import numpy as np
import codecs
from bs4 import BeautifulSoup
from types import GeneratorType
from itertools import tee, cycle
from threading import Thread
import matplotlib.pyplot as plt
from index import IndexIR

ENCODINGS = [ "utf-8", "ISO-8859-1", "IBM037", "IBM039"]

REPLACE_VOWELS = {
    'a': ['ā', 'á', 'ǎ', 'à', 'â'],
    'e': ['é', 'ě', 'è', 'ê'],
    'i': ['í', 'ǐ', 'ì', 'î'],
    'o': ['ó', 'ǒ', 'ò', 'ô'],
    'u': ['ú', 'ü','ǘ', 'ǚ', 'ǔ', 'ǜ', 'ù', 'û'],
    'n': ['ñ']
}

PRE_RE_PATTERNS = {
    'mail': "[a-zA-Z0-9_\-]+@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+",
    'url' : "(?:https:\/\/|http:\/\/){0,1}[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9]+(?:\-[a-zA-Z0-9]+){0,1})+(?:\/[a-zA-Z0-9]+(?:\-[a-zA-Z0-9]+){0,1})*(?:\?[a-zA-Z0-9\-\=\+\&]+){0,1}",
    'abbr': "(?=\s)[A-Za-z]\.(?=[A-Za-z0-9]\.)*(?=(?=\,|\?|\s[a-z0-9]))",
    'real': "[0-9]+\.[0-9]+"
}

POST_RE_PATTERNS = {
    'cel' : "[0-9]+(?:\-[0-9]+)+",
    # 'name': '[A-Z][a-z]+(?:(?:\s{1})[A-Z][a-z]+)*',
    'qty' : '[0-9]+',
    'word': '[a-zA-Z]+'
}

METRICS = [ 'cosine', 'scalar_prod' ]
WEIGHT_SCHEMES = [ 'V1', 'V2', 'V3']
DOCUMENT_TFIDF = {
    'V1': lambda tf, df, docs: tf * log(docs/df,2),
    'V2': lambda tf, df, docs: 1 + log(tf,2),
    'V3': lambda tf, df, docs: (1 + log(tf,2)) * log(docs/df,2)
}
QUERY_TFIDF = {
    'V1': lambda tf, df, max_i, docs: (0.5 + 0.5*tf/max_i*tf) * log(docs/df,2),
    'V2': lambda tf, df, max_i, docs: 1 + log(tf,2),
    'V3': lambda tf, df, max_i, docs: (1 + log(tf,2)) * log(docs/df,2)
}

TMP_RESULT = 'results'
TMP_TERMS = 'terms.json'
TMP_STATS = 'stats.json'
TMP_DOCS_ID = 'docs_id.json'
TMP_TERMS_ID = 'terms_id.json'
TMP_OVERHD = 'overhead.json'


class Tokenizer(object):
    """Docstring for Tokenizer."""

    def __init__(self, dir=getcwd(), queries_file=None, stopwords_file=None, stemmer=None, term_min_len=3, term_max_len=50, verbose=False, weight_scheme='V3', metric='cosine'):
        super(Tokenizer, self).__init__()
        self._dir = dir
        self._stopwords = self.parse_regex(stopwords_file) if (stopwords_file is not None) else None
        self._raw_queries = self.load_queries(queries_file) if (queries_file is not None) else None
        self._queries = dict()
        self._term_min_len = term_min_len
        self._term_max_len = term_max_len
        self._hash_dir = str(sum([ord(i) for i in list(realpath(dir))]))
        if (not exists(self._hash_dir)):
            mkdir(self._hash_dir)
        self._index = IndexIR(self._hash_dir)
        self._docs_id = dict()
        self._terms_id = dict()
        self._qty_docs = 0
        self._qty_tokens = 0
        self._min_tokens_file = inf
        self._max_tokens_file = 0
        self._min_terms_file = inf
        self._max_terms_file = 0
        self._verbose = verbose
        self._replace_vowels = REPLACE_VOWELS
        self._pre_re_patterns = PRE_RE_PATTERNS
        self._post_re_patterns = POST_RE_PATTERNS
        self._stemmer = None
        self._qty_queries = 0
        self._encodings = ENCODINGS
        self._overhead = dict()
        self._done_task = False
        self.tmp_result = join(self._hash_dir, TMP_RESULT)
        if (metric not in METRICS):
            raise ValueError(f" [!] Metric MUST be some of these values {METRICS}")
        self._metric = metric
        if (weight_scheme not in WEIGHT_SCHEMES):
            raise ValueError(f" [!] Weight scheme MUST be some of these values {WEIGHT_SCHEMES}")
        self._weight_scheme = weight_scheme
        self.doc_weight_scheme = lambda tf, df: DOCUMENT_TFIDF[weight_scheme](tf, df, self._qty_docs)
        self.query_weight_scheme = lambda tf, df, max_i : QUERY_TFIDF[weight_scheme](tf, df, max_i, self._qty_docs)
        if ('lancaster' == stemmer):
            self._stemmer = LancasterStemmer()
        elif ('porter' == stemmer):
            self._stemmer = PorterStemmer()
     
        
    def progressbar(self, it, prefix="", size=80, file=stdout):
        if (self._verbose):
            size = size - len(prefix)
            if (isinstance(it, GeneratorType)):
                it, it_backup = tee(it)
                count = sum(1 for x in it_backup)
            else:
                count = len(it)
            write_file = lambda s : [file.write(s), file.flush()] if (self._verbose and prefix != "") else None
            show_progressbar = lambda j : write_file("%s[%s%s] %i/%i\r" % (prefix, "#"*int(size*j/count), "."*(size-int(size*j/count)), j, count))
            if (count > 0):
                show_progressbar(0)
                if (isinstance(it, GeneratorType)):
                    i = 0
                    for item in it:
                        yield item
                        show_progressbar(i+1)
                        i += 1
                else:
                    for i, item in enumerate(it):
                        yield item
                        show_progressbar(i+1) 
                write_file("\n")
        else:
            for item in it:
                yield item


    def animate(self, phrase="", file=stdout):
        if (self._verbose):
            for c in cycle(['.  ', '.. ', '...']):
                if self._done_task:
                    break
                file.write("\r [+] " +  phrase + c)
                file.flush()
                sleep(0.1)
            file.write("\r [+] Done.          \n")


    def start_loading(self, phrase, func, f):
        self._done_task = False
        t = Thread(target=self.animate, args=[phrase])
        t.start()
        if (f == None):
            r = func()
        else:
            r = func(f)
        self._done_task = True
        return r


    def parse_regex(self, file):
        with open(file, 'r') as myfile:
            p = myfile.readlines()
            return '|'.join([r'\b' + i.replace('\n','') + r'\b' for i in p])


    def dump_json(self, obj, file, indent=4):
        with open(file, 'w+') as myfile:
            dump(obj, myfile, indent=indent, ensure_ascii=False)


    def load_json(self, file):
        with open(file, 'r') as myfile:
            return load(myfile)


    def read_file(self, filepath):
        for e in self._encodings:
            try:
                p = codecs.open(filepath, 'r', encoding=e)
                p.readline()
                p.seek(0)
            except UnicodeDecodeError:
                pass
            else:
                return p


    def load_queries(self, filepath):
        if (filepath.lower().endswith('.json')):
            return self.load_json(filepath)
        elif (filepath.lower().endswith('.txt')):
            with open(filepath, "r") as f:
                return {i: q for i, q in enumerate(f.readlines())}
        else:
            raise ValueError("Unsupported file Query format - Only JSON and TXT")


    def normalize_str(self, term):
        """Message"""
        for k in self._replace_vowels:
            for r in self._replace_vowels[k]:
                term = term.replace(r, k)
        return term


    def is_stopword(self, term):
        return True if (self._stopwords != None and match(self._stopwords, term)) else False


    def extract_term(self, term, filepath):
        if (len(term) >= self._term_min_len and len(term) <= self._term_max_len) and (not self.is_stopword(term)):
            id_doc = self._docs_id[filepath]
            self._index.add_entry(term, id_doc)
            return True
        else:
            return False


    def match_list_re(self, patterns, tmpstr, progessbar_str=""):
        tokens = []
        for pattern in patterns:
            if (tmpstr == ''): break
            tokens_parciales = findall(patterns[pattern], tmpstr)
            for i in range(len(tokens_parciales)):
                if (pattern != 'word'):
                    tmpstr = tmpstr.replace(tokens_parciales[i], '#')
                tokens_parciales[i] = tokens_parciales[i].replace('\n', ' ')
            if (pattern == 'word' and self._stemmer is not None):
                tokens_parciales = [self._stemmer.stem(t) for t in tokens_parciales ]
            tokens += tokens_parciales
        return tokens, tmpstr


    def extract_txt_from_html(self, lines):
        soup = BeautifulSoup(lines,  'html.parser')
        return soup.text.strip()


    def remove_ambiguious_tokens(self, raw_html):
        raw = self.extract_txt_from_html(raw_html)
        lowercase_firstletter = lambda s: s[0].lower() + s[1:] if len(s) > 1 else s[0].lower()
        raw = self.normalize_str(raw)
        tokens, raw = self.match_list_re(self._pre_re_patterns, raw)
        sentences = split(r'(?:\.|\?|\!)\s*', raw)
        sentences = split('\n', raw) if len(sentences) == 1 else [lowercase_firstletter(s) for s in sentences if len(s) > 0]
        return [token.lower() for token in tokens], sentences


    def tokenize_sentence(self, sentence):
        tokens, _ = self.match_list_re(self._post_re_patterns, sentence)
        return [token.lower() for token in tokens]


    def is_binary(self, file):
        #textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        #bytes = open(file, 'rb').read(10)
        #return bool(bytes.translate(None, textchars))
        return False


    def tokenize_file(self, filepath):
        if (not self.is_binary(filepath)):
            p = open(filepath, "r")
            tokens, sentences = self.remove_ambiguious_tokens(p.read())
            # print(f" [+]  Matching post RE ({self._post_re_patterns.keys()})...") if (self._verbose) else None
            # for sentence in self.progressbar( sentences, f" [+] {relpath(filepath)}: "):
            self._docs_id[filepath] = self._qty_docs
            for sentence in sentences:
                tokens += self.tokenize_sentence(sentence)
                for token in tokens:
                    self.extract_term(token, filepath)
                # Check maxs & mins - (tokens ands terms)
                self._qty_tokens += len(tokens)
                tokens = []
            p.close()
            self._qty_docs += 1

  
    def process_queries(self):
        for id_query, query in self._raw_queries.items():
            idq = int(id_query)
            tokens, sentences = self.remove_ambiguious_tokens(query)
            for sentence in sentences:
                tokens += self.tokenize_sentence(sentence)
                self._queries[idq] = dict()
                for token in tokens:
                    if (len(token) >= self._term_min_len and len(token) <= self._term_max_len) and (not self.is_stopword(token)):
                        self._queries[idq][token] = 1 if (token not in self._queries[idq]) else self._queries[idq][token]+1
            self._qty_queries += 1


    def calc_query_weights(self, query_terms):
        qty_terms = len(query_terms.keys())
        weights = dict()
        max_i = max(query_terms.values())
        for term in query_terms:
            df = self._index._vocabulary[term]['df'] if term in self._index._vocabulary else 0.0001
            weights[term] = self.query_weight_scheme(
                query_terms[term],
                max_i,
                df
            )
        return weights


    def topk_rank(self, query, k=10):
        """
        TopK rank using a TAAT (Term-At-A-Time) estrategy
        """
        query_terms = list(query.keys())
        weights_query = self.calc_query_weights(query)
        weights_docs = dict()
        postings = { term: self._index.get_plist(term) for term in query_terms }
        scores = dict()
        for term in query_terms:
            df = len(postings[term])
            for doc_id, tf in postings[term]:
                if (doc_id not in scores):
                    scores[doc_id] = 0
                weights_docs[term] = self.doc_weight_scheme(tf, df)
                scores[doc_id] += weights_query[term] * weights_docs[term]
        if (self._metric == 'cosine'):
            for doc_id in scores:
                scores[doc_id] /= (sqrt(sum([wd**2 for wd in weights_docs.values()])) + sqrt(sum([wq**2 for wq in weights_query.values()])))
        
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        heap = dict()
        for i in sorted_scores:
            heap[i[0]] = i[1]
            k -= 1
            if (k == 0):
                break
        return heap


    def make_queries(self):
        if (self._raw_queries is not None):
            def fun():
                self.process_queries()
                if (self._qty_queries > 0):
                    for idq in self._queries:
                        start = time()
                        self._queries[idq]["topk"] = self.topk_rank(self._queries[idq])
                        self._queries[idq]["time"] = time() - start
                    if (self._verbose):
                        print(" [*] Query - Average Time: ",  np.average([self._queries[idq]["time"] for idq in self._queries]))
                    self.get_queries()     
        self.start_loading("Quering", fun, None)


    def get_ranking(self):
        self.process_queries()
        self.make_queries()


    def discovery_dir(self, path=None):
        if (not self._index.index_exists()):
            path = self._dir if (path == None) else path
            start = time()
            for path, _, files in self.progressbar(walk(path), f" [+] {relpath(path)}: "):
                for f in files:
                    if (f == ".DS_Store"):
                        continue
                    fpath = join(path, f)
                    self.tokenize_file(fpath)
            if (self._verbose):
                print(f" [+] Indexing took {time() - start} secs")
            self._index.indexing_ready()
            if (not exists(self.tmp_result)):
                mkdir(self.tmp_result)
            self.get_docs_id()
        else:
            if (self._verbose):
                print(f" [+] Index loaded from {self._dir}.\n [!] Note: delete {self._dir} to force index generation.")
            self.load_docs_id()


    def get_docs_id(self, to_file=TMP_DOCS_ID):
        self.dump_json(self._docs_id, join(self.tmp_result,to_file))

    
    def load_docs_id(self, to_file=TMP_DOCS_ID):
        self._docs_id = self.load_json(join(self.tmp_result,to_file))
        self._qty_docs = len(self._docs_id.keys())
        

    def get_queries(self, to_file='queries'):
        to_file += f"-{self._metric}-{self._weight_scheme}-{datetime.isoformat(datetime.now()).split('.')[0]}.json"
        self.dump_json(self._queries, join(self.tmp_result,to_file.lower()))