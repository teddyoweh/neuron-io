from collections import defaultdict
import re 
import ctypes
from neuio.tools import RE_PATTERNS

class Array(ctypes.Structure):
    _fields_ = [("values", ctypes.POINTER(ctypes.c_char_p)), ("length", ctypes.c_int)]
class CNode(ctypes.Structure):
    pass
CNode._fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_int), ("next", ctypes.POINTER(CNode))]
class CHashMap(ctypes.Structure):
    _fields_ = [("table", ctypes.POINTER(ctypes.POINTER(CNode)))]

class BuildVectors:
    def __init__(self):
        SHARED_LIB = './neuio/extensions/vector.so'
        self.lib = ctypes.CDLL(SHARED_LIB)
        self.lib.c_fit.argtypes = [Array]
        self.lib.c_fit.restype = ctypes.c_int
        self.hash_map_ptr = ctypes.byref(self.lib.VocabStore)
    @property
    def vocab_store_size(self):
        return len(self.vocab_store)
    @property
    def vocab_store(self):
        vocab_dict = {}
        hash_map = ctypes.cast(self.hash_map_ptr, ctypes.POINTER(CHashMap)).contents
        for i in range(1000):
            current = hash_map.table[i]
            while current:
                key = ctypes.string_at(current.contents.key).decode()
                value = current.contents.value
                vocab_dict[key] = value
                current = current.contents.next
        return vocab_dict
    def _clean_texts_(self,texts:list[str]):
        return [re.sub(RE_PATTERNS['non_alphanumeric'],'',_) for _ in texts]
    def transform(self,texts:list):
        texts = self._clean_texts_(texts)
        vectors = []
        for text in texts:
            vector = [0]*self.vocab_store_size
            words = text.split()
            for word in words:
                if word in self.vocab_store:
                    vector_index = list(self.vocab_store.keys()).index(word)
                    vector[vector_index]+=1
            vectors.append(vector)
        return vectors
    def vectorize(self,texts:list):
        texts = self._clean_texts_(texts)
        c_strings = [ctypes.c_char_p(s.encode()) for s in texts]
        c_arr = Array((ctypes.c_char_p * len(c_strings))(*c_strings), len(c_strings))
        result = self.lib.c_vectorize(c_arr)
        return result
        
    def _fit_vectorize(self,texts:list):
        self.vocab_store = defaultdict(int)
        cleaned_texts = self._clean_texts_(texts)
        for _ in cleaned_texts:
            words = _.split()
            for word in words:
                self.vocab_store[word]+=1
        self.vocab_store_size  = len(self.vocab_store)
        return self.transform(cleaned_texts)
    def fit(self,texts:list):
        texts = self._clean_texts_(texts)
        c_strings = [ctypes.c_char_p(s.encode()) for s in texts]
        c_arr = Array((ctypes.c_char_p * len(c_strings))(*c_strings), len(c_strings))
        result = self.lib.c_fit(c_arr)
        return result
 

 
 
  
