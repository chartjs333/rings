"""Standard-library ctypes wrapper; scores are frame-major (count x 10)."""
import ctypes as ct
from pathlib import Path
from native import FrameBuffer,U8P,D64P,INTP

class MulticlassResults:
    def __init__(self,capacity):
        if capacity<0:raise ValueError('Negative capacity')
        self.capacity=capacity
        self.scores=(ct.c_double*(capacity*10))()
        self.labels=(ct.c_int*capacity)()

class MulticlassClassifier:
    def __init__(self,feature_model,readout):
        self.handle=None
        self.lib=lib=ct.CDLL(str(Path(__file__).with_name('libmulticlass.so').resolve()))
        lib.mc_create.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_char_p,ct.c_size_t];lib.mc_create.restype=ct.c_void_p
        lib.mc_destroy.argtypes=[ct.c_void_p];lib.mc_destroy.restype=None
        lib.mc_error.argtypes=[ct.c_void_p];lib.mc_error.restype=ct.c_char_p
        lib.mc_reset.argtypes=[ct.c_void_p];lib.mc_reset.restype=ct.c_int
        lib.mc_process.argtypes=[ct.c_void_p,U8P,ct.c_size_t,ct.c_size_t,ct.c_size_t,ct.c_int,D64P,ct.c_size_t,INTP,ct.c_size_t]
        lib.mc_process.restype=ct.c_int
        error=ct.create_string_buffer(1024)
        self.handle=lib.mc_create(str(Path(feature_model).resolve()).encode(),str(Path(readout).resolve()).encode(),error,len(error))
        if not self.handle:raise ValueError(error.value.decode())
    def _check(self,status):
        if status:raise RuntimeError(self.lib.mc_error(self.handle).decode())
    def reset(self):self._check(self.lib.mc_reset(self.handle))
    def predict_into(self,frames,output,reset_each=True):
        if not isinstance(reset_each,bool):raise ValueError('reset_each must be bool')
        self._check(self.lib.mc_process(self.handle,frames.pointer,frames.nbytes,frames.count,frames.stride,
                                       int(reset_each),output.scores,len(output.scores),output.labels,output.capacity))
        return output
    def close(self):
        if self.handle:self.lib.mc_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
