#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Tokenizer 2
"""

from os import listdir, getcwd, mkdir
from os.path import join, isdir, relpath, exists, basename
from json import load, dump
from sys import stdout
import argparse
from re import findall, split
import codecs
import numpy as np

TRAIN_DIR = '.train-data'

ENCODINGS = [ 'utf-8', "ISO-8859-1"]

REPLACE_VOWELS = {
    'a': ['ā', 'á', 'ǎ', 'à', 'â', 'ã', 'ä'],
    'e': ['é', 'ě', 'è', 'ê', 'ë'],
    'i': ['í', 'ǐ', 'ì', 'î', 'ï'],
    'o': ['ó', 'ǒ', 'ò', 'ô', 'ö'],
    'u': ['ú', 'ü','ǘ', 'ǚ', 'ǔ', 'ǜ', 'ù', 'û'],
    'n': ['ñ'],
    'ss': ['ß'],
    'c': ['ç'],
    ' ': ['\u00ad']
}

class LangDetector(object):
    """Docstring for LangDetector."""

    def __init__(self, dir=getcwd(), train=None, verbose=False, results=None):
        self._dir = dir
        self._verbose = verbose
        self._train = train
        self._letter_dict = dict()
        self._results = results
        self._encodings = ENCODINGS
        self._replace_vowels = REPLACE_VOWELS


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


    def dump_file(self, obj, file):
        with open(file, 'w') as myfile:
            dump(obj, myfile, indent=4)


    def load_file(self, file):
        filepath = basename(file)
        self._letter_dict[filepath] = dict()
        with open(file, 'r') as myfile:
            self._letter_dict[filepath] = load(myfile)


    def read_file(self, filepath):
        for e in self._encodings:
            try:
                p = codecs.open(filepath, 'r', encoding=e)
                p.readlines()
                p.seek(0)
            except UnicodeDecodeError:
                pass
            else:
                return p


    def normalize_str(self, term):
        term = term.lower()
        for k in self._replace_vowels:
            for r in self._replace_vowels[k]:
                term = term.replace(r, k)
        return term


    def process_line(self, line):
        line = self.normalize_str(line)
        return findall('(?![0-9\s\s\»\{\}\[\]\(\)\?\!\¿\¡\'\"\;\,\/\\\|\#\=\+\-\+\*\.\:\&\%\<\>\º]).{1}', line)


    def extract_letter_freq(self, path):
        p = self.read_file(path)
        filepath = basename(path)
        self._letter_dict[filepath] = dict()
        add_one = lambda k, d : 1 if k not in d else d[k] + 1
        for line in self.progressbar( p.readlines(), f" [+] Learning {relpath(filepath)}: "):
            letters = self.process_line(line)
            for letter in letters:
                self._letter_dict[filepath][letter] = add_one(letter, self._letter_dict[filepath])
        # Normalize frequencies
        p.seek(0)
        filesize = len(p.read())
        self._letter_dict[filepath] = { f: i/filesize for f, i in self._letter_dict[filepath].items() }
        self._letter_dict[filepath] = { i[0]: i[1] for i in sorted(self._letter_dict[filepath].items(), key=lambda kv: kv[0]) }
        p.close()


    def predict_sentence(self, filepath):
        p = self.read_file(filepath)
        add_one  = lambda k, d : 1 if k not in d else d[k] + 1
        langd = []
        sizefile = len(p.read())
        p.seek(0)
        for line in self.progressbar( p.readlines(), f" [+] Predicting {relpath(filepath)}: "):
            distrib = {}
            letters = self.process_line(line)
            for letter in letters:
                distrib[letter] = add_one(letter, distrib)
            score = {}
            distrib = { f: i/sizefile for f, i in distrib.items() }
            for lang in self._letter_dict:
                for k in self._letter_dict[lang]:
                    if (k not in distrib):
                        distrib[k] = 0
                distrib = { i[0]: i[1] for i in sorted(distrib.items(), key=lambda kv: kv[0]) }
                score[lang] = np.corrcoef(list(self._letter_dict[lang].values()), list(distrib.values()))[1][0]
            langd.append(max(score, key=score.get))
        self.dump_file(langd, "results.json")
        p.close()
        if (self._results):
            self.validate_results(langd, self._results)
        return lang


    def validate_results(self, res, expectedfile):
        ex = self.read_file(expectedfile)
        expected = ex.readlines()
        correct = 0
        for i in range(len(res)):
            expected_r = findall('(?![0-9\s]).+', expected[i])[0].lower()
            if (res[i].lower() == expected_r):
                print(f" [+] {i} - {res[i].lower()} OK")
                correct += 1
            else:
                print(f" [!] {i} - {res[i].lower()} - Expected: {expected_r} FAILED")
        print(f" {'-'*40}\n\tResults : {correct}/{len(res)}")
        print(f" \tAccuracy: {correct/len(res)}\n")
        ex.close()


    def train(self):
        if (exists(TRAIN_DIR)):
            self.discovery_dir(TRAIN_DIR, self.load_file)
        elif (self._train is not None):
            self.discovery_dir(self._train, self.extract_letter_freq)
            mkdir(TRAIN_DIR)
            for k, i in self._letter_dict.items():
                self.dump_file(i, join(TRAIN_DIR, k))
        else:
            print("No training files were provided! Please run with -h/--help option")


    def predict(self):
        if (self._dir):
            if (exists(self._dir)):
                self.discovery_dir(self._dir, self.predict_sentence)
        else:
            print("No predict files were provided! Please run with -h/--help option")


    def discovery_dir(self, path=None, method=None, args=[]):
        path = self._dir if (path == None) else path
        if isdir(path):
            files = listdir(path)
            print(f"\nScanning {path}...\n") if (self._verbose) else None
            for f in files:
                self.discovery_dir(join(path, f), method, args)
        else:
            method(path, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, default=None, help="files or directory to train models, default: None")
    parser.add_argument('-p', '--predict', type=str, default=None, help="files or directory to predict language, default: None")
    parser.add_argument('-r', '--results', type=str, default=None, help="files or directory that contains the expected result, default: None")
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help="show messages during process")
    args = parser.parse_args()
    l = LangDetector(dir=args.predict, train=args.train, verbose=args.verbose, results=args.results)
    l.train()
    l.predict()


if __name__ == '__main__':
    main()
