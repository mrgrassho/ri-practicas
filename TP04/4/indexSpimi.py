#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import mkdir, remove
from os.path import join, exists, isfile
from struct import pack, unpack, calcsize
from json import dump, load
from io import BytesIO

INDEXIR_DIR = '.index'
INDEXIR_INV = 'inverted_index.bin'
VOCABULARY = 'vocabulary.json'

class IndexIR(object):
    """Vocabulary and Inverted Index Data Structures for Information Retrieval"""
    
    def __init__(self, top_dir="", index_on_memory=False, qty_partial_docs=400):
        self._vocabulary = dict()
        self._tmp_vocabulary = dict()
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
        self._qty_partial_docs = qty_partial_docs
        self._processed_docs = 0
        self._block = 0
        self._prev_id_doc = -1
        self._terms_set = set({})
        if (self.index_exists()):
            self._vocabulary = self._load_vocab()


    def _pack_plist(self, plist):
        data = [e for entries in plist for e in entries]
        return pack(f"{len(data)}I", *data)


    def _unpack_plist(self, plist):
        data = unpack(f"{len(plist)//self._int_sz}I", plist)
        return [(data[i],data[i+1]) for i in range(0,len(data),2)]


    def _is_memory_available(self):
        if (self._processed_docs == self._qty_partial_docs):
            self._processed_docs = 0
            return False
        else:
            return True


    def add_entry(self, term, id_doc):
        if (self._is_memory_available()):
            if (not term in self._tmp_vocabulary):
                self._terms_set.add(term)
                self._tmp_vocabulary[term] = {"df": 0, "plist": []}
            plist = self._tmp_vocabulary[term]['plist']
            tup = (id_doc, 1)
            indx = None
            for i in range(len(plist)):
                if (plist[i][0] == id_doc):
                    indx = i
                    break
            if (indx is not None):
                tup = (plist[indx][0], plist[indx][1]+1)
                self._tmp_vocabulary[term]['plist'][indx] = tup
            else:
                self._tmp_vocabulary[term]['plist'].append(tup)
                self._tmp_vocabulary[term]['df'] += 1
            
            if (id_doc != self._prev_id_doc):
                self._prev_id_doc = id_doc
                self._processed_docs = (id_doc + 1) - self._block * self._qty_partial_docs
        else:
            self._tmp_vocabulary = self.dump_index(self._tmp_vocabulary, join(self._parent_dir, f"block-{self._block}.bin"))
            for term in self._tmp_vocabulary.keys():
                self._tmp_vocabulary[term].pop("plist")
            self.dump_json(self._tmp_vocabulary, join(self._parent_dir, f"block-{self._block}.json"), None, (',', ':'))
            self._block += 1
            self._tmp_vocabulary = dict()


    def dump_index(self, voc, file):
        with open(file, 'ab') as f:
            for term in voc:
                packed_tups = self._pack_plist(voc[term]['plist'])
                f.seek(0, 2) # SEEK_END
                voc[term]['offset'] = f.tell()
                f.write(packed_tups)
        return voc


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
            dump(data, f, indent=None, ensure_ascii=False, separators=(',',':'))


    def _load_index(self):
        with open(self._index_fpath, "rb") as f:
            self._index_buffer = f.read()


    def _dump_index(self):
        with open(self._index_fpath, "wb") as f:
            f.write(self._index_buffer.getvalue())


    def _merge_blokcs(self):
        voc = []
        for i in range(self._block):
            voc.append(self.load_json(join(self._parent_dir, f"block-{i}.json")))
        with open(self._index_fpath, 'wb') as fvocab:
            for term in self._terms_set:
                full_unpacked_list = []
                for i in range(len(voc)):                
                    index_fpath = join(self._parent_dir, f"block-{i}.bin") 
                    with open(index_fpath, "rb") as f:
                        if (term in voc[i]):
                            f.seek(voc[i][term]['offset'])
                            packed_list = f.read(voc[i][term]['df'] * self._plist_item_sz)
                            full_unpacked_list += self._unpack_plist(packed_list)
                    fvocab.seek(0, 2) # SEEK_END
                    self._vocabulary[term] = dict()
                    self._vocabulary[term]['offset'] = fvocab.tell()
                    self._vocabulary[term]['df'] = len(full_unpacked_list)
                    fvocab.write(self._pack_plist(full_unpacked_list))
        self._dump_vocab()
        self._remove_blocks()


    def _remove_blocks(self):
        for i in range(self._block):
            remove(join(self._parent_dir, f"block-{i}.json"))
            remove(join(self._parent_dir, f"block-{i}.bin"))


    def indexing_ready(self):
        if (len(self._tmp_vocabulary) > 0):
            self.dump_index(self._tmp_vocabulary, join(self._parent_dir, f"block-{self._block}.bin"))
            self.dump_json(self._tmp_vocabulary, join(self._parent_dir, f"block-{self._block}.json"),None, (',', ':'))
            self._block += 1
            self._tmp_vocabulary = dict()
        self._merge_blokcs()

    
    def index_exists(self):
        return isfile(self._index_fpath)


    def dump_json(self, obj, to_file, indent=4, separators=(', ', ': ')):
        sorted_data = sorted(obj.items(), key=lambda kv: kv[0])
        data = { i[0]: i[1] for i in sorted_data}
        with open(to_file, 'w') as f:
            dump(data, f, indent=indent, ensure_ascii=False, separators=separators)


    def load_json(self, to_file):
        with open(to_file, 'r') as f:
            return load(f)