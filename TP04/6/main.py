#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from os import getcwd
from tokenizer import Tokenizer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--min', type=int, default=3, help="Minimun length of terms")
    parser.add_argument('-x', '--max', type=int, default=50, help="Maximun length of terms")
    parser.add_argument('-s', '--stopwords', type=str, help="File cotaining stopwords")
    parser.add_argument('-q', '--queries', type=str, help="File containing the queries. \nAccepted Formats:\n(TXT: some text to query\\n...), (JSON: { 'idq': 'some text to query', ... })")
    parser.add_argument('-d', '--dir', type=str, default=getcwd(), help="Directory to scan, default: current working dir")
    parser.add_argument('-nv', '--not-verbose', action='store_false', default=True, help="Show no messages during process")
    parser.add_argument('-t', '--stemmer', choices=["lancaster","porter"], help="Choose stemmer")
    parser.add_argument('-m', '--metric', choices=["cosine","scalar_prod"], default="cosine", help="Choose a metric to used during raking.")
    parser.add_argument('-w', '--weight', choices=["V1","V2","V3"], default="V3", help="Choose weighting scheme")
    args = parser.parse_args()
    t = Tokenizer(dir=args.dir, queries_file=args.queries, stopwords_file=args.stopwords, stemmer=args.stemmer, term_min_len=args.min, term_max_len=args.max, verbose=args.not_verbose, metric=args.metric, weight_scheme=args.weight)
    t.discovery_dir()
    t.make_queries()

if __name__ == '__main__':
    main()