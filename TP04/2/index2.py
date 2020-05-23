#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import mkdir
from os.path import join, exists, isfile
from struct import pack, unpack, calcsize
from json import dump, load
from io import BytesIO
from collections import OrderedDict

INDEXIR_DIR = '.index'
INDEXIR_INV = 'inverted_index.bin'
VOCABULARY = 'vocabulary.json'

class IndexIR(object):
    """Vocabulary and Inverted Index Data Structures for Information Retrieval"""
    
    def __init__(self, top_dir="", index_on_memory=False):
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
        self._int_sz = calcsize(f"I")
        self._plist_item_sz = self._int_sz * 2
        if (self.index_exists()):
            self._vocabulary = self._load_vocab()


    def _update_offsets(self, term, qty=1):
        keys = list(self._vocabulary.keys())
        update_from_term = keys.index(term) + 1
        for i in range(update_from_term, len(keys)):
            k = keys[i]
            self._vocabulary[k]['offset'] += self._plist_item_sz * qty


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
            indx = None
            for i in range(len(plist)):
                if (plist[i][0] == id_doc):
                    indx = i
                    break
            
            if (indx is not None):
                tup = [(plist[indx][0], plist[indx][1]+1)]
                offset = self._vocabulary[term]['offset'] + indx * self._plist_item_sz
            else:
                new_entry = True
                offset = self._vocabulary[term]['offset'] + self._vocabulary[term]['df'] * self._plist_item_sz
        self._add_plist(term, tup, offset, new_entry)


    def _add_plist(self, term, tup, offset, new_entry):
        packed_tup = self._pack_plist(tup)
        if (term in self._vocabulary):
            if (new_entry):
                if (self._vocabulary[term]['allocated_spaces_left'] == 0):
                    self._index_buffer.seek(offset)
                    prev_data = self._index_buffer.read()
                    allocate_qty = self._vocabulary[term]['df'] * 2
                    self._vocabulary[term]['allocated_spaces_left'] = allocate_qty
                    self._update_offsets(term, allocate_qty)
                    self._index_buffer.seek(offset + self._plist_item_sz * allocate_qty)
                    self._index_buffer.write(prev_data)
                    self._index_buffer.seek(offset)
                    self._index_buffer.write(packed_tup)
                else:
                    self._index_buffer.seek(offset)
                    self._index_buffer.write(packed_tup)
                self._vocabulary[term]['allocated_spaces_left'] -= 1
                self._vocabulary[term]['df'] += 1
            else:
                self._index_buffer.seek(offset)
                self._index_buffer.write(packed_tup)     
        else:
            self._index_buffer.seek(0, 2) # SEEK_END
            self._vocabulary[term] = {
                'df': 1,
                'offset': self._index_buffer.tell(),
                'allocated_spaces_left': 0
            }
            self._index_buffer.write(packed_tup)


    def get_plist(self, term):
        if (term in self._vocabulary):
            if (self._in_memory):
                f = self._index_buffer 
                f.seek(self._vocabulary[term]['offset'])
                packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz)
                return self._unpack_plist(packed_list)
            else:
                with open(self._index_fpath, "rb") as f:
                    f.seek(self._vocabulary[term]['offset'])
                    packed_list = f.read(self._vocabulary[term]['df'] * self._plist_item_sz)
                    return self._unpack_plist(packed_list)
        else:
            return []

    
    def _load_vocab(self):
        with open(self._vocab_fpath, 'r') as myfile:
            return load(myfile)


    def _dump_vocab(self):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[0])
        data = { i[0]: i[1] for i in sorted_data}
        with open(self._vocab_fpath, 'w') as f:
            dump(data, f, indent=2, ensure_ascii=False)


    def _load_index(self):
        with open(self._index_fpath, "rb") as f:
            self._index_buffer = f.read()


    def _dump_index(self):
        with open(self._index_fpath, "wb") as f:
            f.write(self._index_buffer.getvalue())


    def indexing_ready(self):
        self._dump_vocab()
        self._dump_index()

    
    def index_exists(self):
        return isfile(self._index_fpath)


    def dump_json(self, to_file, indent=4):
        sorted_data = sorted(self._vocabulary.items(), key=lambda kv: kv[0])
        data = { i[0]: i[1] for i in sorted_data}
        with open(to_file, 'w') as f:
            dump(data, f, indent=indent, ensure_ascii=False)
