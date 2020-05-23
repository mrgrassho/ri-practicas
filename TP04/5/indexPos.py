#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join, exists, isfile
from struct import pack, unpack, calcsize
from json import dump, load
from io import BytesIO
from collections import OrderedDict
from math import fabs

INDEXIR_DIR = '.index'
INDEXIR_INV = 'inverted_index.bin'
VOCABULARY = 'vocabulary.json'

class IndexIR(object):
    """Vocabulary and Inverted Index Data Structures for Information Retrieval"""
    
    def __init__(self, top_dir=""):
        self._vocabulary = OrderedDict()
        self._parent_dir = join(top_dir,INDEXIR_DIR)
        if (not exists(self._parent_dir)):
            mkdir(self._parent_dir)
        self._index_fpath = join(self._parent_dir, INDEXIR_INV)
        if (not exists(self._index_fpath)):
            # Si no existe el index creamos uno en memoria
            self._index_buffer = BytesIO(b"")
        self._indexing_ready = False
        self._vocab_fpath = join(self._parent_dir, VOCABULARY)
        self._int_sz = calcsize(f"I")
        self._plist_item_sz = self._int_sz * 2
        self._doc_position = dict()
        if (self.index_exists()):
            self._vocabulary = self._load_vocab()


    def _pack_plist(self, plist):
        data = []
        for pl in plist:
            data.append(pl[0])
            data.append(pl[1])
            for e in pl[2]:
                data.append(e)
        return pack(f"{len(data)}I", *data)


    def _unpack_plist(self, plist):
        data = unpack(f"{len(plist)//self._int_sz}I", plist)
        udata = [] 
        i = 0
        while (i < len(data)):
            entry = []
            entry.append(data[i])
            i += 1
            len_pos = data[i]
            entry.append(len_pos)
            i += 1
            positions = []
            for j in range(i,i+len_pos):
                positions.append(data[j])
            i += len_pos
            entry.append(positions)
            udata.append(entry)
        return udata

    
    def _get_next_postion(self, id_doc):
        postion = 0
        if (id_doc not in self._doc_position):
            self._doc_position[id_doc] = 0
        else:
            self._doc_position[id_doc] += 1
            postion = self._doc_position[id_doc]
        return postion


    def add_entry(self, term, id_doc):
        if (not term in self._vocabulary):
            self._vocabulary[term] = {"df": 0, "tf": 0, "plist": []}
        plist = self._vocabulary[term]['plist']
        pos = self._get_next_postion(id_doc)
        tup = [id_doc, 1, [pos]]
        indx = None
        for i in range(len(plist)):
            if (plist[i][0] == id_doc):
                indx = i
                break
        if (indx is not None):
            tup = [plist[indx][0], plist[indx][1]+1, plist[indx][2] + [pos] ]
            self._vocabulary[term]['plist'][indx] = tup
        else:
            self._vocabulary[term]['plist'].append(tup)
            self._vocabulary[term]['df'] += 1
        self._vocabulary[term]['tf'] += 1


    def _flush_index(self):
        for term in self._vocabulary:
            packed_tups = self._pack_plist(self._vocabulary[term]['plist'])
            self._index_buffer.seek(0, 2) # SEEK_END
            self._vocabulary[term]['offset'] = self._index_buffer.tell()
            self._index_buffer.write(packed_tups)


    def docs_intersection(self, terms, k=1):
        plist_prev = self.get_plist(terms[0])
        docs_prev = [ pl[0] for pl in plist_prev ]
        positions_prev = [ pl[2] for pl in plist_prev ]
        docs_result = []
        for term in terms[1:]:
            plist = self.get_plist(term)
            for doc_id, _, positions in plist:
                if (doc_id in docs_prev):
                    index = docs_prev.index(doc_id)
                    flag = False
                    for p1 in positions_prev[index]:
                        for p2 in positions:
                            if (fabs(p1 - p2) <= k):
                                flag = True
                                break
                    if (flag):
                        docs_result.append(doc_id)
        return docs_result


    def get_plist(self, term):
        if (term in self._vocabulary):
            if (not self.indexing_ready):
                self._index_buffer.seek(self._vocabulary[term]['offset'])
                packed_list = self._index_buffer.read(self._vocabulary[term]['df'] * self._plist_item_sz + self._vocabulary[term]['tf'] * self._int_sz)
                return self._unpack_plist(packed_list)
            else:
                with open(self._index_fpath, "rb") as f:
                    f.seek(self._vocabulary[term]['offset'])
                    packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz + self._vocabulary[term]['tf'] * self._int_sz)
                    return self._unpack_plist(packed_list)
        else:
            return []

    
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
        self._indexing_ready = True
        self._flush_index()
        self._dump_vocab()
        self._dump_index()

    
    def index_exists(self):
        return isfile(self._index_fpath)


    def dump_json(self, to_file, indent=4):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[0])
        data = { i[0]: i[1] for i in sorted_data}
        with open(to_file, 'w') as f:
            dump(data, f, indent=indent, ensure_ascii=False)
        return data
