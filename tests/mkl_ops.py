import numpy as np
import ctypes
from ctypes import c_int, c_float, POINTER, CDLL
import os
import time

# Load MKL
mkl = ctypes.cdll.LoadLibrary("/opt/intel/oneapi/mkl/2024.2/lib/libmkl_rt.so")

CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANS = 111

mkl.cblas_sgemv.restype = None
mkl.cblas_sgemv.argtypes = [
    ctypes.c_int,  # layout
    ctypes.c_int,  # trans
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

def mkl_dense_run(weight, input_vec, bias, n_runs=1000):
    M, N = weight.shape
    A = np.ascontiguousarray(weight.astype(np.float32))
    x = np.ascontiguousarray(input_vec.astype(np.float32))
    y = np.ascontiguousarray(bias.astype(np.float32))  # output

    start = time.perf_counter()
    for _ in range(n_runs):
        y[:] = bias.astype(np.float32)
        mkl.cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            M,
            N,
            ctypes.c_float(1.0),
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            N,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1,
            ctypes.c_float(1.0),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            1
        )
    end = time.perf_counter()
    avg_time_ms = (end - start) * 1000 / n_runs
    return y, avg_time_ms