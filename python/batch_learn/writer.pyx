from libcpp.string cimport string
from libcpp cimport bool

import numpy as np
cimport numpy as np

cdef extern from "../../include/batch_learn.hpp" namespace "batch_learn":
    cdef cppclass file_writer:
        file_writer(string, int)
        void write_row(int, int*, int*, float*, float, int)
        void write_index()

cdef class Writer:
    cdef file_writer* _writer
    cdef bool _closed

    def __cinit__(self, filename, int index_bits):
        self._writer = new file_writer(filename.encode('utf-8'), index_bits)
        self._closed = False

    def __dealloc__(self):
        if not self._closed:
            self.close()

    def write_index(self):
        self._check_not_closed()
        self._writer.write_index()

    def write_row(self, fields, indices, values, y, group=0):
        self._check_not_closed()
        self._write_row(
            np.asarray(fields, dtype=np.uint32, order='c'),
            np.asarray(indices, dtype=np.uint32, order='c'),
            np.asarray(values, dtype=np.float32, order='c'),
            y,
            group
        )

    def close(self):
        self._check_not_closed()
        del self._writer
        self._closed = True

    def _check_not_closed(self):
        if self._closed:
            raise RuntimeError("Writer already closed")

    def _write_row(
        self,
        np.ndarray[uint, ndim=1, mode='c'] fields,
        np.ndarray[uint, ndim=1, mode='c'] indices,
        np.ndarray[float, ndim=1, mode='c'] values,
        float y,
        int group
    ):
        self._writer.write_row(fields.shape[0], <int*> fields.data, <int*> indices.data, <float*> values.data, y, group)
