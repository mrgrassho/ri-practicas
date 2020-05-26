#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from os import getcwd
from tokenizer import Tokenizer 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--min', type=int, default=3, help="minimun length of terms")
    parser.add_argument('-x', '--max', type=int, default=50, help="maximun length of terms")
    parser.add_argument('-s', '--stopwords', type=str, help="file containing stopwords")
    parser.add_argument('-q', '--queries', type=str, help="file containing the queries")
    parser.add_argument('-d', '--dir', type=str, default=getcwd(), help="directory to scan, default: current working dir")
    parser.add_argument('-v', '--verbose', action='store_false', default=True, help="Show messages during process")
    parser.add_argument('-t', '--stemmer', choices=["lancaster","porter"], default=True, help="choose stemmer")
    parser.add_argument('-i', '--in-memory-index', action='store_true', default=False, help="keep index in memory – loads `.index.bin` to memory")
    parser.add_argument('-sk', '--skip-pointers', action='store_true', default=False, help="Use skip pointers")
    args = parser.parse_args()
    t = Tokenizer(dir=args.dir, queries_file=args.queries, stopwords_file=args.stopwords, stemmer=args.stemmer, term_min_len=args.min, term_max_len=args.max, verbose=args.verbose, in_memory_index=args.in_memory_index, skip_pointers=args.skip_pointers)
    t.discovery_dir()
    t.make_queries()


if __name__ == '__main__':
    main()