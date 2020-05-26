#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join, exists, isfile
from struct import pack, unpack, calcsize
from json import dump, load
from io import BytesIO
from collections import OrderedDict
from math import sqrt, ceil

INDEXIR_DIR = '.index'
INDEXIR_INV = 'inverted_index.bin'
VOCABULARY = 'vocabulary.json'
CONFIG_DIR = 'config.json'

class IndexIR(object):
    """Vocabulary and Inverted Index Data Structures for Information Retrieval"""
    
    def __init__(self, top_dir="", index_on_memory=False, skip_pointers=False):
        self._vocabulary = OrderedDict()
        self._parent_dir = join(top_dir,INDEXIR_DIR)
        if (not exists(self._parent_dir)):
            mkdir(self._parent_dir)
        self._index_fpath = join(self._parent_dir, INDEXIR_INV)
        if (not exists(self._index_fpath)):
            # Si no existe el index creamos uno en memoria
            self._index_buffer = BytesIO(b"")
            self._in_memory = True
        else:
            # Si existe tomamos el index cargado de disco
            if (index_on_memory):
                self._load_index()
            self._in_memory = index_on_memory
        self._vocab_fpath = join(self._parent_dir, VOCABULARY)
        self._config_fpath = join(self._parent_dir, CONFIG_DIR)
        self._int_sz = calcsize(f"I")
        self._plist_item_sz = self._int_sz * 2
        self._config = {
            "SKIP_POINTERS": skip_pointers
        }
        if (self.index_exists()):
            self._vocabulary = self._load_vocab()
            self._config = self._load_conf()


    def _pack_plist(self, plist):
        data = [e for entries in plist for e in entries]
        return pack(f"{len(data)}I", *data)


    def _calculate_len_plist(self, len_packed_data):
        l = len_packed_data
        a = 4
        b = -4*(1+l)-1 
        c = (1+l)**2
        # Las listas con skip_pointers responden a la siguiente función
        f = lambda x: 2*x + sqrt(x) - 1
        s1 = ((-1*b) + sqrt(b**2 - 4*a*c))/(2*a)
        s2 = ((-1*b) - sqrt(b**2 - 4*a*c))/(2*a)
        if (f(s1) == l):
            return ceil(s1)
        elif (f(s2) == l):
            return ceil(s2)


    def _unpack_plist(self, plist):
        is_skip = lambda i, l: i % (l//int(sqrt(l) - 1)) == 0 and (i + (l//int(sqrt(l) - 1))) < l if int(sqrt(l) - 1) > 0 else False
        data = unpack(f"{len(plist)//self._int_sz}I", plist)
        res = []
        if (self._config['SKIP_POINTERS']):
            i = 0
            len_d = self._calculate_len_plist(len(data))
            j = 0
            while i < len(data)-1:
                entry = []
                entry.append(data[i])
                entry.append(data[i+1])
                if (is_skip(j, len_d)):
                    entry.append(data[i+2])
                    i += 1
                i += 2
                j += 1
                res.append(entry)
        else:
            res = [(data[i],data[i+1]) for i in range(0,len(data),2)]
        return res


    def _add_skip_pointers(self):
        if (exists(self._index_fpath)):
            raise Exception(f"An index alredy exists. Delete the folder {self._index_fpath}")
        prev_skip = lambda i, l: (i - (l//int(sqrt(l) - 1)))
        is_prev_skip = lambda i, l: (i % (l//int(sqrt(l) - 1)) == 0 and i != 0) if int(sqrt(l) - 1) > 0 else False
        if (self._config['SKIP_POINTERS']):
            for term in self._vocabulary:
                l = len(self._vocabulary[term]['plist'])
                for i in range(len(self._vocabulary[term]['plist'])):
                    if (is_prev_skip(i, l)):
                        pi = prev_skip(i, l)
                        doc_id = self._vocabulary[term]['plist'][pi][0]
                        tf = self._vocabulary[term]['plist'][pi][1]
                        self._vocabulary[term]['plist'][pi] = (doc_id, tf, i)


    def add_entry(self, term, id_doc):
        if (not term in self._vocabulary):
            self._vocabulary[term] = {"df": 0, "tf": 0, "plist": []}
        plist = self._vocabulary[term]['plist']
        tup = (id_doc, 1)
        indx = None
        for i in range(len(plist)):
            if (plist[i][0] == id_doc):
                indx = i
                break
        if (indx is not None):
            tup = (plist[indx][0], plist[indx][1]+1)
            self._vocabulary[term]['plist'][indx] = tup
        else:
            self._vocabulary[term]['plist'].append(tup)
            self._vocabulary[term]['df'] += 1
        self._vocabulary[term]['tf'] += 1


    def flush_index(self):
        for term in self._vocabulary:
            packed_tups = self._pack_plist(self._vocabulary[term]['plist'])
            self._index_buffer.seek(0, 2) # SEEK_END
            self._vocabulary[term]['offset'] = self._index_buffer.tell()
            self._index_buffer.write(packed_tups)


    def get_plist(self, term):
        if (term in self._vocabulary):
            if (self._in_memory):
                self._index_buffer.seek(self._vocabulary[term]['offset'])
                if (self._config['SKIP_POINTERS']):
                    packed_list = self._index_buffer.read(self._vocabulary[term]['df'] * self._plist_item_sz + self._int_sz * int(sqrt(self._vocabulary[term]['df']) - 1))
                else:
                    packed_list = self._index_buffer.read(self._vocabulary[term]['df'] * self._plist_item_sz)
                return self._unpack_plist(packed_list)
            else:
                with open(self._index_fpath, "rb") as f:
                    f.seek(self._vocabulary[term]['offset'])
                    if (self._config['SKIP_POINTERS']):
                        packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz + self._int_sz * int(sqrt(self._vocabulary[term]['df']) - 1))
                    else:
                        packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz)
                    return self._unpack_plist(packed_list)
        else:
            return []


    def get_plist_intersect(self, t1, t2):
        result = { term: [] for term in [t1,t2] }
        p1 = self.get_plist(t1)
        p2 = self.get_plist(t2)
        i1 = 0
        i2 = 0
        has_skip = lambda i, l: len(l[i]) == 3
        next_skip = lambda i, l: l[i][2]
        while (i1 < len(p1) and i2 < len(p2)):
            if p1[i1][0] == p2[i2][0]:
                result[t1] += [p1[i1]]
                result[t2] += [p2[i2]]
                i1 += 1
                i2 += 1
            elif p1[i1][0] < p2[i2][0]:
                f = False
                while has_skip(i1,p1):
                    if p1[next_skip(i1,p1)][0] < p2[i2][0]:
                        i1 = next_skip(i1,p1)
                        f = True
                    else:
                        if (not f):
                            i1 += 1
                        break
                else:  
                    i1 += 1
            else:
                f = False
                while has_skip(i2,p2):
                    if p2[next_skip(i2,p2)][0] < p1[i1][0]:
                        i2 = next_skip(i2,p2)
                        f = True
                    else:
                        if (not f):
                            i2 += 1
                        break
                else:  
                    i2 += 1
        return result.values()


    def _load_conf(self):
        with open(self._config_fpath, 'r') as myfile:
            return load(myfile)

    
    def _dump_conf(self):
        with open(self._config_fpath, 'w') as f:
            dump(self._config, f, indent=4, ensure_ascii=False)


    def _load_vocab(self):
        with open(self._vocab_fpath, 'r') as myfile:
            return load(myfile)


    def _dump_vocab(self):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[0])
        data = { i[0]: { "df": i[1]["df"], "tf": i[1]["tf"], "offset": i[1]["offset"] } for i in sorted_data}
        with open(self._vocab_fpath, 'w') as f:
            dump(data, f, indent=None, separators=(',',':'), ensure_ascii=False)


    def _load_index(self):
        with open(self._index_fpath, "rb") as f:
            self._index_buffer = BytesIO(b"")
            self._index_buffer.write(f.read())


    def _dump_index(self):
        with open(self._index_fpath, "wb") as f:
            f.write(self._index_buffer.getvalue())


    def indexing_ready(self):
        self._add_skip_pointers()
        self.flush_index()
        self._dump_vocab()
        self._dump_index()
        self._dump_conf()

    
    def index_exists(self):
        return isfile(self._index_fpath) and isfile(self._vocab_fpath) and isfile(self._config_fpath)
                


    def dump_json(self, to_file, indent=4):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[0])
        data = { i[0]: i[1] for i in sorted_data}
        with open(to_file, 'w') as f:
            dump(data, f, indent=indent, ensure_ascii=False)
