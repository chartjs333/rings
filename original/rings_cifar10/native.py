"""C++ runtime wrapper: standard library only; reusable zero-copy batch buffers."""
import ctypes as ct
from pathlib import Path

IMAGE_BYTES = 3072
U8P = ct.POINTER(ct.c_uint8)
D64P = ct.POINTER(ct.c_double)
INTP = ct.POINTER(ct.c_int)


class FrameBuffer:
    """Pin a writable contiguous buffer; immutable bytes are copied once at setup.

    Frames: interleaved 32x32 RGB uint8, optional trailing padding per frame.
    Do not mutate this buffer while C++ is reading it (ctypes releases the GIL).
    """
    def __init__(self, data, *, count=None, stride=IMAGE_BYTES):
        if not isinstance(stride, int) or stride < IMAGE_BYTES:
            raise ValueError("stride must be at least 3072 bytes")
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("Expected a C-contiguous buffer")
        view = view.cast("B")
        self.copied = view.readonly
        if self.copied:
            view = memoryview(bytearray(view))
        if count is None:
            if len(view) % stride:
                raise ValueError("Specify count for a partially padded final frame")
            count = len(view) // stride
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be nonnegative")
        required = (count-1)*stride+IMAGE_BYTES if count else 0
        if len(view) < required:
            raise ValueError("Truncated frame buffer")
        self.view, self.count, self.stride = view, count, stride
        self.nbytes = len(view)
        self.array = (ct.c_uint8*self.nbytes).from_buffer(view)
        self.pointer = ct.cast(self.array, U8P)


class Results:
    def __init__(self, capacity):
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self.scores = (ct.c_double*capacity)()
        self.labels = (ct.c_int*capacity)()


class NativeClassifier:
    """One mutable state per handle. Use separate handles in concurrent threads."""
    def __init__(self, model, library=None):
        self.handle = None
        library = Path(library) if library else Path(__file__).with_name("librings.so")
        self.lib = lib = ct.CDLL(str(library.resolve()))
        specs = {
            "rr_create": ([ct.c_char_p, ct.c_char_p, ct.c_size_t], ct.c_void_p),
            "rr_destroy": ([ct.c_void_p], None),
            "rr_error": ([ct.c_void_p], ct.c_char_p),
            "rr_reset": ([ct.c_void_p], ct.c_int),
            "rr_feature_count": ([ct.c_void_p], ct.c_size_t),
            "rr_predict": ([ct.c_void_p,U8P,ct.c_size_t,ct.c_int,D64P,INTP],ct.c_int),
            "rr_process_frames": ([ct.c_void_p,U8P,ct.c_size_t,ct.c_size_t,ct.c_size_t,
                                    ct.c_int,D64P,ct.c_size_t,INTP,ct.c_size_t],ct.c_int),
            "rr_extract_features": ([ct.c_void_p,U8P,ct.c_size_t,ct.c_size_t,ct.c_size_t,
                                      ct.c_int,D64P,ct.c_size_t],ct.c_int),
        }
        for name, (args, result) in specs.items():
            getattr(lib,name).argtypes=args
            getattr(lib,name).restype=result
        error = ct.create_string_buffer(1024)
        self.handle = lib.rr_create(str(Path(model).resolve()).encode(),error,len(error))
        if not self.handle:
            raise ValueError(error.value.decode())
        self.feature_count = lib.rr_feature_count(self.handle)

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.rr_error(self.handle).decode())

    def predict_into(self, frames, results, *, reset_each=True):
        """One native call; no frame copies, output allocations or list conversion."""
        if not isinstance(reset_each, bool):
            raise ValueError("reset_each must be bool")
        self._check(self.lib.rr_process_frames(
            self.handle, frames.pointer, frames.nbytes, frames.count, frames.stride,
            int(reset_each), results.scores, results.capacity, results.labels, results.capacity))
        return results

    def features_into(self, frames, output, *, reset_each=True):
        """Write features to a caller-owned ctypes double array; one native call."""
        if not isinstance(reset_each, bool):
            raise ValueError("reset_each must be bool")
        self._check(self.lib.rr_extract_features(
            self.handle,frames.pointer,frames.nbytes,frames.count,frames.stride,
            int(reset_each),output,len(output)))

    def reset(self):
        self._check(self.lib.rr_reset(self.handle))

    def close(self):
        if self.handle:
            self.lib.rr_destroy(self.handle)
            self.handle = None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
