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
    'mail': '[a-zA-Z0-9_\-]+@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+',
    'url' : '(?:https:\/\/|http:\/\/){0,1}[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9]+(?:\-[a-zA-Z0-9]+){0,1})+(?:\/[a-zA-Z0-9]+(?:\-[a-zA-Z0-9]+){0,1})*(?:\?[a-zA-Z0-9\-\=\+\&]+){0,1}',
    'abbr': '(?=\s)[A-Za-z]\.(?=[A-Za-z0-9]\.)*(?=(?=\,|\?|\s[a-z0-9]))',
    'real': '[0-9]+\.[0-9]+'
}

POST_RE_PATTERNS = {
    'cel' : '[0-9]+(?:\-[0-9]+)+',
    'name': '[A-Z][a-z]+(?:(?:\s{1})[A-Z][a-z]+)*',
    'qty' : '[0-9]+',
    'word': '[a-zA-Z]+'
}

TMP_RESULT = 'results'
TMP_TERMS = 'terms.json'
TMP_STATS = 'stats.json'
TMP_DOCS_ID = 'docs_id.json'
TMP_TERMS_ID = 'terms_id.json'
TMP_TERMS_DISTRIB = 'terms_distrib.png'
TMP_OVERHD = 'overhead.json'
TMP_OVERHD_DOC = 'overhead_by_doc.png'
TMP_OVERHD_TOTAL = 'overhead.png'


class Tokenizer(object):
    """Docstring for Tokenizer."""

    def __init__(self, dir=getcwd(), queries_file=None, stopwords_file=None, stemmer=None, term_min_len=3, term_max_len=50, verbose=False):
        super(Tokenizer, self).__init__()
        self._dir = dir
        self._stopwords = self.parse_regex(stopwords_file) if (stopwords_file is not None) else None
        self._raw_queries = self.load_json(queries_file) if (queries_file is not None) else None
        self._queries = dict()
        self._term_min_len = term_min_len
        self._term_max_len = term_max_len
        self._index = IndexIR()
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
        if ('lancaster' == stemmer):
            self._stemmer = LancasterStemmer()
        elif ('porter' == stemmer):
            self._stemmer = PorterStemmer()
     
        
    def progressbar(self, it, prefix="", size=80, file=stdout):
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


    def animate(self, phrase="", file=stdout):
        for c in cycle(['.  ', '.. ', '...']):
            if self._done_task:
                break
            file.write("\r" +  phrase + 'loading' + c)
            file.flush()
            sleep(0.1)
        file.write("\r" +  phrase + 'loaded.\n')


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
        textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        bytes = open(file, 'rb').read(1024)
        return bool(bytes.translate(None, textchars))


    def tokenize_file(self, filepath):
        is_greater = lambda x, y: x if (x > y) else y
        is_less = lambda x, y: x if (x < y) else y
        total_terms = 0
        if (not self.is_binary(filepath)):
            p = open(filepath, "r")
            tokens, sentences = self.remove_ambiguious_tokens(p.read())
            # print(f" [+]  Matching post RE ({self._post_re_patterns.keys()})...") if (self._verbose) else None
            # for sentence in self.progressbar( sentences, f" [+] {relpath(filepath)}: "):
            self._docs_id[filepath] = self._qty_docs
            for sentence in sentences:
                tokens += self.tokenize_sentence(sentence)
                terms = 0
                new_terms = 0
                for token in tokens:
                    qty_elem_before = self._index._vocabulary[token]['df'] if token in self._index._vocabulary else 0
                    if (self.extract_term(token, filepath)):
                        terms += 1
                        if qty_elem_before < self._index._vocabulary[token]['df']:
                            new_terms += 1
                total_terms += new_terms
                # Check maxs & mins - (tokens ands terms)
                self._max_tokens_file = is_greater( len(tokens), self._max_tokens_file)
                self._min_tokens_file = is_less( len(tokens), self._min_tokens_file)
                self._max_terms_file = is_greater( terms, self._max_terms_file)
                self._min_terms_file = is_less( terms, self._min_terms_file)
                self._qty_tokens += len(tokens)
                tokens = []
            p.close()
            self._qty_docs += 1
        return total_terms


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


    def discovery_dir(self, path=None):
        path = self._dir if (path == None) else path
        for path, _, files in walk(path):
            for f in self.progressbar(files, f" [+] {relpath(path)}: "):
                fpath = join(path, f)
                new_qty_terms = self.tokenize_file(fpath)
                sz_index = new_qty_terms*8
                sz_doc = getsize(fpath)
                self._overhead[self._docs_id[fpath]] = {
                    "doc_sz": sz_doc, # in bytes
                    "doc_index_sz": sz_index, # in bytes
                    "percentage_overhead": round(sz_index/sz_doc,6)
                }
        if (not exists(TMP_RESULT)):
            mkdir(TMP_RESULT)
        self.get_stats()
        self.get_docs_id()
        self.get_overhead()
        self.get_terms()        
        self._index.indexing_ready()


    def get_docs_id(self, to_file=TMP_DOCS_ID):
        self.dump_json(self._docs_id, join(TMP_RESULT,to_file))
        

    def get_terms(self, to_file=TMP_TERMS):
        data = self._index.dump_json(join(TMP_RESULT,to_file))
        # Plot Overhead By Doc
        width = 1
        fig, ax = plt.subplots()
        a = [ i[1]['df']*8 for i in data.items()]
        ax.bar(range(len(data.keys())), a, width, label="Doc_i Size")
        ax.set_ylabel("Size in Bytes")
        ax.set_xlabel("Terms")
        ax.set_title("Plist Size By Term")
        ax.legend()
        plt.savefig(join(TMP_RESULT,TMP_TERMS_DISTRIB))


    def get_overhead(self, to_file=TMP_OVERHD):
        self.dump_json(self._overhead, join(TMP_RESULT,to_file))
        width = 0.35
        # Plot Overhead By Doc
        fig, ax = plt.subplots()
        a = [ i[1]['doc_sz'] for i in self._overhead.items()]
        b = [ i[1]['doc_index_sz'] for i in self._overhead.items()]
        c = [ i[1]['percentage_overhead']*100 for i in self._overhead.items()]
        bar1 = ax.bar(self._overhead.keys(), a, width, label="Doc_i Size")
        bar2 = ax.bar(self._overhead.keys(), b, width, label="Doc_i Index Size")
        ax.set_ylabel("Size in Bytes")
        ax.set_xlabel("Documents")
        ax.set_title("Doc_i Size vs Doc_i Index Size")
        for i in range(len(bar1)):
            height = bar2[i].get_height()
            plt.text(bar1[i].get_x() + bar1[i].get_width()/2.0, height, f'{c[i]:.1f}%', ha='center', va='bottom')
        ax.legend()
        plt.savefig(join(TMP_RESULT,TMP_OVERHD_DOC))
        # Plot Overhead all docs
        width = 0.5
        fig, ax = plt.subplots()
        bar1 = ax.bar([self._dir], [sum(a)], width, label="Collection Size")
        bar2 = ax.bar([self._dir], [sum(b)], width, label="Index Size")
        ax.set_ylabel("Size in Bytes")
        ax.set_xlabel("Collection")
        ax.set_title("Collection Size vs Index Size")
        height = bar2[0].get_height()
        plt.text(bar1[0].get_x() + bar1[0].get_width()/2.0, height, f'{(sum(b)/sum(a))*100:.2f}%', ha='center', va='bottom')
        ax.legend()
        plt.savefig(join(TMP_RESULT,TMP_OVERHD_TOTAL))
        return self._overhead


    def get_stats(self, to_file=TMP_STATS):
        qty_terms = len(self._index._vocabulary.keys())
        stats = {
            "qty_docs": self._qty_docs,
            "qty_tokens": self._qty_tokens,
            "qty_terms": qty_terms,
            "avg_qty_tokens": self._qty_tokens / self._qty_docs,
            "avg_qty_terms": qty_terms / self._qty_docs,
            "avg_len_term": mean([len(t) for t in self._index._vocabulary.keys()]),
            "min_tokens_file": self._min_tokens_file,
            "max_tokens_file": self._max_tokens_file,
            "min_terms_file": self._min_terms_file,
            "max_terms_file": self._max_terms_file,
            # "qty_terms_one_appeareance": len([t for t in self._terms if sum([i[1] for i in self._terms[t][1]]) == 1])
        }
        self.dump_json(stats, join(TMP_RESULT,to_file))
        return stats