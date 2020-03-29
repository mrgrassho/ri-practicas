#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Tokenizer 4
"""

from os import listdir, getcwd
from os.path import join, isdir, relpath
from json import load, dump
from statistics import mean
from math import inf
from sys import stdout, getsizeof
import argparse
from re import match, split, findall
from nltk.stem import SnowballStemmer


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
    'name': '[A-Z][a-z]+(?:(?:\s*|\s*\n\s*)[A-Z][a-z]+)*',
    'qty' : '[0-9]+',
    'word': '[a-zA-Z]+'
}

class Tokenizer(object):
    """Docstring for Tokenizer."""

    def __init__(self, dir=getcwd(), stopwords_file=None, term_min_len=3, term_max_len=50, verbose=False):
        super(Tokenizer, self).__init__()
        self._dir = dir
        self._stopwords = self.parse_regex(stopwords_file) if (stopwords_file is not None) else None
        self._term_min_len = term_min_len
        self._term_max_len = term_max_len
        self._terms = dict()
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
        self._stemmer = SnowballStemmer("spanish")


    def progressbar(self, it, prefix="", size=80, file=stdout):
        size = size - len(prefix)
        count = len(it)
        write_file = lambda s : [file.write(s), file.flush()] if (self._verbose and prefix != "") else None
        show_progressbar = lambda j : write_file("%s[%s%s] %i/%i\r" % (prefix, "#"*int(size*j/count), "."*(size-int(size*j/count)), j, count))
        if (count > 0):
            show_progressbar(0)
            for i, item in enumerate(it):
                yield item
                show_progressbar(i+1)
            write_file("\n")


    def parse_regex(self, file):
        with open(file, 'r') as myfile:
            p = myfile.readlines()
            return '|'.join([r'\b' + i.replace('\n','') + r'\b' for i in p])


    def dump_file(self, obj, file):
        with open(file, 'w') as myfile:
            dump(obj, myfile, indent=4)


    def normalize_str(self, term):
        for k in self._replace_vowels:
            for r in self._replace_vowels[k]:
                term = term.replace(r, k)
        return term


    def is_stopword(self, term):
        return True if (match(self._stopwords, term)) else False


    def extract_term(self, term, filepath):

        if (len(term) >= self._term_min_len and len(term) <= self._term_max_len) and (not self.is_stopword(term)):
            if (term in self._terms):
                self._terms[term]['all'] += 1
                if (filepath in self._terms[term]):
                    self._terms[term][filepath] += 1
                else:
                    self._terms[term]['df'] += 1
                    self._terms[term][filepath] = 1
            else:
                self._terms[term] = {'all': 1, 'df': 1}
                self._terms[term][filepath] = 1
            return True
        else:
            return False


    def match_list_re(self, patterns, str, progessbar_str=""):
        tokens = []
        for pattern in self.progressbar(patterns, prefix=progessbar_str) :
            if (str == ''): break
            tokens_parciales = findall(patterns[pattern], str)
            for token in tokens_parciales:
                if (pattern != 'word'):
                    str = str.replace(token, '#')
            if (pattern == 'word'):
                tokens_parciales = [self._stemmer.stem(t) for t in tokens_parciales]
            tokens += tokens_parciales
        return tokens, str


    def remove_ambiguious_tokens(self, raw):
        lowercase_firstletter = lambda s: s[0].lower() + s[1:] if len(s) > 1 else s[0].lower()
        raw = self.normalize_str(raw)
        tokens, raw = self.match_list_re(self._pre_re_patterns, raw)
        sentences = split('(?:\.|\?|\!)\s*', raw)
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
            for sentence in self.progressbar( sentences, f" [+] {relpath(filepath)}: "):
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
            self._qty_docs += 1
        else:
            print(f" [+] {relpath(filepath)}: is being ignored (binary file)") if (self._verbose) else None


    def discovery_dir(self, path=None):
        path = self._dir if (path == None) else path
        if isdir(path):
            files = listdir(path)
            print(f"\nScanning {path}...\n") if (self._verbose) else None
            for f in files:
                self.discovery_dir(join(path, f))
        else:
            self.tokenize_file(path)


    def get_terms(self, to_file='terms.json'):
        sorted_data = sorted(self._terms.items(), key=lambda kv: kv[1]['all'], reverse=True)
        data = { i[0]: i[1] for i in sorted_data}
        self.dump_file(data, to_file)
        return data


    def get_stats(self, to_file='stats.json'):
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
            "qty_terms_one_appeareance": len([t for t in self._terms if self._terms[t]["all"] == 1])
        }
        self.dump_file(stats, to_file)
        return stats


    def get_freq(self, to_file='frecuencies.json'):
        freq = {
            'least frequent terms': { i[0]: i[1]['all'] for i in sorted(self._terms.items(), key=lambda kv: kv[1]['all'])[:11]},
            'most  frequent terms': { i[0]: i[1]['all'] for i in sorted(self._terms.items(), key=lambda kv: kv[1]['all'], reverse=True)[:11]}
        }
        self.dump_file(freq, to_file)
        return freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--min', type=int, default=3, help="minimun length of terms")
    parser.add_argument('-x', '--max', type=int, default=50, help="maximun length of terms")
    parser.add_argument('-s', '--stopwords', type=str, help="file containing stopwords")
    parser.add_argument('-d', '--dir', type=str, default=getcwd(), help="directory to scan, default: current working dir")
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help="show messages during process")
    args = parser.parse_args()
    t = Tokenizer(dir=args.dir, stopwords_file=args.stopwords, term_min_len=args.min, term_max_len=args.max, verbose=args.verbose)
    t.discovery_dir()
    t.get_stats()
    t.get_freq()
    t.get_terms()


if __name__ == '__main__':
    main()
