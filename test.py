import ctypes
import pandas as pd 
import re
lib = ctypes.CDLL("./test.so")
class Array(ctypes.Structure):
    _fields_ = [("values", ctypes.POINTER(ctypes.c_char_p)), ("length", ctypes.c_int)]
lib.c_fit.argtypes = [Array]
lib.c_fit.restype = ctypes.c_int

def c_fit_python(arr):
    c_strings = [ctypes.c_char_p(s.encode()) for s in arr]
    c_arr = Array((ctypes.c_char_p * len(c_strings))(*c_strings), len(c_strings))
    result = lib.c_fit(c_arr)
    return result
class CNode(ctypes.Structure):
    pass

CNode._fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_int), ("next", ctypes.POINTER(CNode))]

class CHashMap(ctypes.Structure):
    _fields_ = [("table", ctypes.POINTER(ctypes.POINTER(CNode)))]
def vocab_store_to_dict(hash_map_ptr):
    vocab_dict = {}
    hash_map = ctypes.cast(hash_map_ptr, ctypes.POINTER(CHashMap)).contents
    for i in range(1000):
        current = hash_map.table[i]
        while current:
            key = ctypes.string_at(current.contents.key).decode()
            value = current.contents.value
            vocab_dict[key] = value
            current = current.contents.next
    return vocab_dict
df = pd.read_csv('./twitter_training.csv')
df = df.dropna()
text_data = list(df['text'])
arr = [re.sub(r'[^a-zA-Z0-9\s]','',_) for _ in text_data]


if __name__ == "__main__":
    result = c_fit_python(arr)
    print("C function returned:", result)
    vocab_dict = vocab_store_to_dict(ctypes.byref(lib.VocabStore))
    print(vocab_dict)
