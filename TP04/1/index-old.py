#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import mkdir, remove
from os.path import join, exists
from struct import pack, unpack, calcsize
from collections import OrderedDict
from json import dumps

INDEXRI_DIR = '.index'
INDEXRI_INV = 'inverted_index.bin'

class IndexRI(object):
    """Vocabulary and Inverted Index Data Structures for information retrieval"""
    
    def __init__(self):
        self._vocabulary = OrderedDict()
        if (not exists(INDEXRI_DIR)):
            mkdir(INDEXRI_DIR)
        self._index_fpath = join(INDEXRI_DIR, INDEXRI_INV)
        self._int_sz = calcsize(f"I")
        self._plist_item_sz = self._int_sz * 2
        if (exists(self._index_fpath)):
            remove(self._index_fpath)


    def _update_offsets(self, term):
        keys = list(self._vocabulary.keys())
        update_from_term = keys.index(term) + 1
        for i in range(update_from_term, len(keys)):
            k = keys[i]
            self._vocabulary[k]['offset'] += self._plist_item_sz


    def _pack_plist(self, plist):
        data = [e for entries in plist for e in entries]
        return pack(f"{len(data)}I", *data)


    def _unpack_plist(self, plist):
        data = unpack(f"{len(plist)//self._int_sz}I", plist)
        return [(data[i],data[i+1]) for i in range(0,len(data),2)]


    def add_entry(self, term, id_doc):
        plist = self.get_plist(term)
        tup = [(id_doc, 1)]
        offset = 0
        new_entry = False
        if (len(plist) > 0):
            tmplist = [plist[i][0] for i in range(len(plist))]
            if (id_doc in tmplist):
                i = tmplist.index(id_doc)
                tup = [(plist[i][0], plist[i][1]+1)]
                plist[i] = tup[0]
                offset = self._vocabulary[term]['offset'] + i * self._plist_item_sz
            else:
                plist.append(tup[0])
                new_entry = True
                offset = self._vocabulary[term]['offset'] + self._vocabulary[term]['df'] * self._plist_item_sz
        else:
            plist.append(tup[0])
        self._add_plist(term, tup, offset, new_entry)
        return plist


    def _add_plist(self, term, tup, offset, new_entry):
        packed_tup = self._pack_plist(tup)
        if (exists(self._index_fpath) and term in self._vocabulary):
            with open(self._index_fpath, "rb+") as f:
                if (new_entry):
                    f.seek(offset)
                    prev_data = f.read()
                    f.seek(offset)
                    f.write(packed_tup)
                    f.seek(offset + self._plist_item_sz)
                    f.write(prev_data)
                    self._update_offsets(term)
                    self._vocabulary[term]['df'] += 1
                else:
                    f.seek(offset)
                    f.write(packed_tup)     
        else:
            with open(self._index_fpath, "ab+") as f:
                f.seek(0, 2) # SEEK_END
                self._vocabulary[term] = {
                    'df': 1,
                    'offset': f.tell()
                }
                f.write(packed_tup)


    def get_plist(self, term):
        if (term in self._vocabulary):
            with open(self._index_fpath, "rb") as f:
                f.seek(self._vocabulary[term]['offset'])
                packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz)
                return self._unpack_plist(packed_list)
        else:
            return []
    

    def dump_json(self, to_file):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[1]['df'], reverse=True)
        data = { i[0]: i[1] for i in sorted_data}
        with open(to_file, 'w') as f:
            f.seek(0)
            f.write("{\n")
            for k, v in data.items():
                f.write(f"\t\"{k}\": [\n\t\t{v['df']},\n\t\t{dumps(self.get_plist(k))}\n\t],\n")
            f.write("}\n")
