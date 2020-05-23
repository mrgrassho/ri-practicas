#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    name: RI-TP-03.06 
    author: Victorio Scafati
"""

from os import listdir, getcwd, walk, mkdir, chdir, environ
from os.path import join, isdir, realpath, relpath, exists
from json import load, dump, dumps
from statistics import mean
from math import inf, log, sqrt
from sys import stdout, getsizeof
import argparse
from re import match, split, findall
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from time import time, sleep
import numpy as np
import codecs
from bs4 import BeautifulSoup
from types import GeneratorType
from itertools import tee, cycle
from threading import Thread

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

# WEIGHT SCHEMES

DOCUMENT_TFIDF = {
    'V1': lambda tf, df, docs: tf * log(docs/df,2),
    'V2': lambda tf: 1 + log(tf,2),
    'V3': lambda tf, df, docs: (1 + log(tf,2)) * log(docs/df,2)
}

QUERY_TFIDF = {
    'V1': lambda tf, df, max_i, docs: (0.5 + 0.5*tf/max_i*tf) * log(docs/df,2),
    'V2': lambda tf: 1 + log(tf,2),
    'V3': lambda tf, df, docs: (1 + log(tf,2)) * log(docs/df,2)
}


TMP_INDEX = '.index'
TMP_TERMS = 'terms.json'
TMP_STATS = 'stats.json'
TMP_DOCS_ID = 'docs_id.json'
TMP_TERMS_ID = 'terms_id.json'
TMP_DOC_WEIGHTS = 'doc_weights.json'
TMP_TDMATRIX = 'td_matrix'

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
        self._terms = dict()
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
        self._log_terms = []
        self.doc_weight_scheme = lambda tf, df: DOCUMENT_TFIDF['V1'](tf, df, self._qty_docs)
        self.query_weight_scheme = lambda tf, df, max_i : QUERY_TFIDF['V1'](tf, df, max_i, self._qty_docs)
        self._td_matrix = None
        self._qty_queries = 0
        self._encodings = ENCODINGS
        self._hash_dir = str(sum([ord(i) for i in list(realpath(dir))]))
        self._cache_dir = self.create_cache_dir()
        self._loaded_from_cache = self.load_cache()
        self._done_task = True
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


    def dump_json(self, obj, file):
        with open(file, 'w+') as myfile:
            dump(obj, myfile, indent=4, ensure_ascii=False)


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
            if (term in self._terms):
                self._terms[term]['tf'] += 1
                if (filepath in self._terms[term]):
                    self._terms[term][filepath] += 1
                else:
                    self._terms[term]['df'] += 1
                    self._terms[term][filepath] = 1
            else:
                self._terms[term] = {'tf': 1, 'df': 1}
                self._terms[term][filepath] = 1
            self._log_terms.append(len(self._terms))
            return True
        else:
            self._log_terms.append(len(self._terms))
            return False


    def match_list_re(self, patterns, tmpstr, progessbar_str=""):
        tokens = []
        for pattern in patterns:
            if (tmpstr == ''): break
            tokens_parciales = findall(patterns[pattern], tmpstr)
            for token in tokens_parciales:
                token.replace('\n', ' ')
                if (pattern != 'word'):
                    tmpstr = tmpstr.replace(token, '#')
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
        if (not self.is_binary(filepath)):
            p = open(filepath, "r")
            tokens, sentences = self.remove_ambiguious_tokens(p.read())
            # print(f" [+]  Matching post RE ({self._post_re_patterns.keys()})...") if (self._verbose) else None
            # for sentence in self.progressbar( sentences, f" [+] {relpath(filepath)}: "):
            for sentence in sentences:
                tokens += self.tokenize_sentence(sentence)
                terms = len([token for token in tokens if (self.extract_term(token, filepath))])
                # Check maxs & mins - (tokens ands terms)
                self._max_tokens_file = is_greater( len(tokens), self._max_tokens_file)
                self._min_tokens_file = is_less( len(tokens), self._min_tokens_file)
                self._max_terms_file = is_greater( terms, self._max_terms_file)
                self._min_terms_file = is_less( terms, self._min_terms_file)
                self._qty_tokens += len(tokens)
                tokens = []
            p.close()
            self._docs_id[filepath] = self._qty_docs
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


    def get_ranking(self):
        self.process_queries()
        self.calc_query_weights()
        self.vector_model()


    def vector_model(self):
        print(" [+] Building Vector model...")
        #b = [sqrt(np.sum(np.power(self._td_matrix[:,i], 2))) for i in range(self._qty_docs)]
        b = [sqrt(v) for i, v in self._doc_sum_weights.items()]
        ranking = []
        for i in range(self._qty_queries):
            rank_matrix = np.zeros((1, self._qty_docs))
            a = np.sum(np.power(self._td_query[:,i], 2))
            if (a != 0):
                for j in self.progressbar(range(self._qty_docs), f" [+] Query {i}:"):
                    if (b[j] != 0):
                        rank_matrix[0, j] = np.sum(np.multiply(self._td_query[:,i], self._td_matrix[:,j])) /  (sqrt(a) * b[j])
            e = list(enumerate(rank_matrix[0,]))
            e.sort(key=lambda t: t[1], reverse=True)
            top_50 = e[:50]
            doc_names = [list(self._docs_id.keys())[list(self._docs_id.values()).index(i[0])]  for i in top_50]
            ranking += [(i,top_50[j][0], doc_names[j], top_50[j][1]) for j in range(50)]
        with open('rank_matrix.csv', 'w+') as f:
            for item in ranking:
                f.write(f"{item[0]} {item[1]} {item[2]} {item[3]:.4f}\n")


    def calc_query_weights(self):
        print(" [+] Processing queries...")
        qty_terms = len(self._terms.keys())
        self._td_query = np.zeros((qty_terms, self._qty_queries))
        for i, terms in self.progressbar(self._queries.items(), prefix=f" [+] Weighting queries :"):
            max_i = max(self._queries[i].values())
            for term in terms:
                if (term in self._terms):
                    self._td_query[self._terms_id[term],i] = self.query_weight_scheme(
                        self._queries[i][term],
                        max_i,
                        self._terms[term]['df']
                    )
                    

    def calc_doc_weights(self):
        qty_terms = len(self._terms.keys())
        self._td_matrix = np.zeros((qty_terms, self._qty_docs))
        self._doc_sum_weights = { i:0 for i in range(self._qty_docs)}
        for term, docs in self.progressbar(self._terms.items(), prefix=" [+] Weighting docs"):
            for doc in docs:
                if (doc != 'df' and doc != 'tf'):
                    i = self._terms_id[term]
                    j = self._docs_id[doc]
                    self._td_matrix[i, j] = self.doc_weight_scheme(
                        self._terms[term][doc],
                        self._terms[term]['df']
                    )
                    self._doc_sum_weights[j] += (self._terms[term][doc]**2)
        sorted_data = sorted(self._doc_sum_weights.items(), key=lambda kv: kv[0])
        self._doc_sum_weights = { i[0]: i[1] for i in sorted_data}

    
    def discovery_dir(self, path=None):
        if (not self._loaded_from_cache):
            path = self._dir if (path == None) else path
            for path, _, files in self.progressbar(walk(path), f" [+] {relpath(path)}: "):
                for f in files:
                    self.tokenize_file(join(path, f))
            #self.get_stats()
            #self.get_docs_id()
            #self.get_terms_id()
            #self.get_doc_weights()
            #self.get_terms()
            np.savez_compressed(join(self._cache_dir,TMP_TDMATRIX), a=self._td_matrix)


    def get_terms_id(self, to_file=TMP_TERMS_ID):
        self._terms_id = { i[1]: i[0] for i in list(enumerate(self._terms.keys()))}
        self.dump_json(self._terms_id, join(self._cache_dir,to_file))


    def get_doc_weights(self, to_file=TMP_DOC_WEIGHTS):   
        self.calc_doc_weights()
        self.dump_json(self._doc_sum_weights, join(self._cache_dir,to_file))     


    def get_docs_id(self, to_file=TMP_DOCS_ID):
        self.dump_json(self._docs_id, join(self._cache_dir,to_file))
        

    def get_terms(self, to_file=TMP_TERMS):
        sorted_data = sorted(self._terms.items(), key=lambda kv: kv[1]['tf'], reverse=True)
        data = { i[0]: {'df': i[1]['df'], 'tf': i[1]['tf']} for i in sorted_data}
        self.dump_json(data, join(self._cache_dir,to_file))
        return data


    def get_log_terms(self, to_file='log_terms.json'):
        self.dump_json(self._log_terms, to_file)
        return self._log_terms


    def get_stats(self, to_file=TMP_STATS):
        qty_terms = len(self._terms.keys())
        stats = {
            "qty_docs": self._qty_docs,
            "qty_tokens": self._qty_tokens,
            "qty_terms": qty_terms,
            "avg_qty_tokens": self._qty_tokens / self._qty_docs,
            "avg_qty_terms": qty_terms / self._qty_docs,
            "avg_len_term": mean([len(t) for t in self._terms.keys()]),
            "min_tokens_file": self._min_tokens_file,
            "max_tokens_file": self._max_tokens_file,
            "min_terms_file": self._min_terms_file,
            "max_terms_file": self._max_terms_file,
            "qty_terms_one_appeareance": len([t for t in self._terms if self._terms[t]["tf"] == 1])
        }
        self.dump_json(stats, join(self._cache_dir,to_file))
        return stats


    def create_cache_dir(self):
        if (not exists(TMP_INDEX)):
            mkdir(TMP_INDEX)
        tmp_cache = join(TMP_INDEX,  self._hash_dir)
        if (not exists(tmp_cache)):
            mkdir(tmp_cache)
        return tmp_cache


    def load_cache(self):
        loaded = self.load_terms(join(self._cache_dir, TMP_TERMS)) \
            and self.load_stats(join(self._cache_dir,TMP_STATS)) \
            and self.load_docs_id(join(self._cache_dir,TMP_DOCS_ID)) \
            and self.load_terms_id(join(self._cache_dir,TMP_TERMS_ID)) \
            and self.load_doc_weights(join(self._cache_dir,TMP_DOC_WEIGHTS)) \
            and self.load_tdmatrix(join(self._cache_dir,TMP_TDMATRIX + ".npz"))
        self._done_task = True
        if (loaded):
            print(f" [+] Loaded from cache. To force re-run -> Delete '{self._cache_dir}'")
        return loaded


    def start_loading(self, phrase, func, f):
        self._done_task = False
        t = Thread(target=self.animate, args=[phrase])
        t.start()
        r = func(f)
        self._done_task = True
        return r
        

    def load_terms(self, from_file):
        e = exists(from_file)
        if (e): 
            self._terms = self.start_loading(" [+] Terms ", self.load_json, from_file)
        return e


    def load_stats(self, from_file):
        e = exists(from_file)
        if (e):
            obj = self.load_json(from_file)
            self._qty_docs = obj['qty_docs']
            self._qty_tokens = obj['qty_tokens']
            self._min_tokens_file = obj["min_tokens_file"]
            self._max_tokens_file = obj["max_tokens_file"]
            self._min_terms_file = obj["min_terms_file"]
            self._max_terms_file = obj["max_terms_file"]
            print(f" [+] Stats loaded")
        return e


    def load_docs_id(self, from_file):
        e = exists(from_file)
        if (e):
            self._docs_id = self.load_json(from_file)
            print(f" [+] ID Docs loaded.")
        return e


    def load_terms_id(self, from_file):
        e = exists(from_file)
        if (e):
            self._terms_id = self.load_json(from_file)
            print(f" [+] ID Terms loaded.")
        return e


    def load_doc_weights(self, from_file):
        e = exists(from_file)
        if (e):
            self._doc_sum_weights = self.load_json(from_file)
            print(f" [+] Docs Weights loaded.")
        return e

    def load_tdmatrix(self, from_file):
        e = exists(from_file)
        if (e):
            self._td_matrix = self.start_loading(" [+] TD Matrix ", np.load, from_file)['a']
        return e


    def get_freq(self, to_file='frecuencies.json'):
        freq = {
            'least frequent terms': { i[0]: i[1]['tf'] for i in sorted(self._terms.items(), key=lambda kv: kv[1]['tf'])[:11]},
            'most  frequent terms': { i[0]: i[1]['tf'] for i in sorted(self._terms.items(), key=lambda kv: kv[1]['tf'], reverse=True)[:11]}
        }
        self.dump_json(freq, to_file)
        return freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--min', type=int, default=3, help="minimun length of terms")
    parser.add_argument('-x', '--max', type=int, default=50, help="maximun length of terms")
    parser.add_argument('-s', '--stopwords', type=str, help="file containing stopwords")
    parser.add_argument('-q', '--queries', type=str, help="file containing the queries")
    parser.add_argument('-d', '--dir', type=str, default=getcwd(), help="directory to scan, default: current working dir")
    parser.add_argument('-v', '--verbose', action='store_false', default=True, help="Show messages during process")
    parser.add_argument('-t', '--stemmer', choices=["lancaster","porter"], help="choose stemmer")
    args = parser.parse_args()
    t = Tokenizer(dir=args.dir, queries_file=args.queries, stopwords_file=args.stopwords, stemmer=args.stemmer, term_min_len=args.min, term_max_len=args.max, verbose=args.verbose)
    process_time = time()
    t.discovery_dir()
    t.get_ranking()
    print(f" [+] {args.stemmer.capitalize()} process - took {time() - process_time} seconds.")



if __name__ == '__main__':
    main()
