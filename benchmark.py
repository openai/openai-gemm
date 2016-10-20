#!/usr/bin/env python

import numpy as np
import pycuda.driver as drv
from neon.backends.nervanagpu import NervanaGPU

from openai_gemm import matmul


ng = NervanaGPU()
print drv.Context.get_current().get_device().name()

config = (
    #    m,    n,    k, AT,    BT  (row order)
    (   16, 1760, 1760, False, False),
    (   32, 1760, 1760, False, False),
    (   64, 1760, 1760, False, False),
    (  128, 1760, 1760, False, False),
    ( 7000, 1760, 1760, False, False),
    (   16, 2048, 2048, False, False),
    (   32, 2048, 2048, False, False),
    (   64, 2048, 2048, False, False),
    (  128, 2048, 2048, False, False),
    ( 7000, 2048, 2048, False, False),
    (   16, 2560, 2560, False, False),
    (   32, 2560, 2560, False, False),
    (   64, 2560, 2560, False, False),
    (  128, 2560, 2560, False, False),
    ( 7000, 2560, 2560, False, False),
    (   16, 4096, 4096, False, False),
    (   32, 4096, 4096, False, False),
    (   64, 4096, 4096, False, False),
    (  128, 4096, 4096, False, False),
    ( 7000, 4096, 4096, False, False),
    (   16, 1760, 1760, False,  True),
    (   32, 1760, 1760, False,  True),
    (   64, 1760, 1760, False,  True),
    (  128, 1760, 1760, False,  True),
    ( 7000, 1760, 1760, False,  True),
    (   16, 2048, 2048, False,  True),
    (   32, 2048, 2048, False,  True),
    (   64, 2048, 2048, False,  True),
    (  128, 2048, 2048, False,  True),
    ( 7000, 2048, 2048, False,  True),
    (   16, 2560, 2560, False,  True),
    (   32, 2560, 2560, False,  True),
    (   64, 2560, 2560, False,  True),
    (  128, 2560, 2560, False,  True),
    ( 7000, 2560, 2560, False,  True),
    (   16, 4096, 4096, False,  True),
    (   32, 4096, 4096, False,  True),
    (   64, 4096, 4096, False,  True),
    (  128, 4096, 4096, False,  True),
    ( 7000, 4096, 4096, False,  True),
    ( 7133, 1760, 1760, True , False),
    ( 7133, 2048, 2048, True , False),
    ( 7133, 2560, 2560, True , False),
    ( 7133, 4096, 4096, True , False),
    ( 9124, 5124, 1760, False, False),
    ( 9124, 5124, 2048, False, False),
    ( 9124, 5124, 2560, False, False),
    ( 9124, 5124, 4096, False, False),
    ( 9124, 5124, 1760, False,  True),
    ( 9124, 5124, 2048, False,  True),
    ( 9124, 5124, 2560, False,  True),
    ( 9124, 5124, 4096, False,  True),
    ( 8457,   35, 1760, False, False),
    ( 8457,   35, 2048, False, False),
    ( 8457,   35, 2560, False, False),
    ( 8457,   35, 4096, False, False),
    ( 8457,   35, 1760, False,  True),
    ( 8457,   35, 2048, False,  True),
    ( 8457,   35, 2560, False,  True),
    ( 8457,   35, 4096, False,  True),
    (   16, 7680, 2560, False, False),
    (   32, 7680, 2560, False, False),
    (   64, 7680, 2560, False, False),
    (  128, 7680, 2560, False, False),
    (   16, 7680, 2560, False,  True),
    (   32, 7680, 2560, False,  True),
    (   64, 7680, 2560, False,  True),
    (  128, 7680, 2560, False,  True),
    (   16, 3072, 1024, False, False),
    (   32, 3072, 1024, False, False),
    (   64, 3072, 1024, False, False),
    (  128, 3072, 1024, False, False),
    (   16, 3072, 1024, False,  True),
    (   32, 3072, 1024, False,  True),
    (   64, 3072, 1024, False,  True),
    (  128, 3072, 1024, False,  True),
    ( 7435, 3072, 1024, True , False),
    ( 5481, 7680, 2560, True , False),

    # (60000,   32,   32, True , False),
    # (60000,  256,  256, True , False),

    # ( 4096, 4096,   32, True , False),
    # ( 3456, 3456,   32, True , False),
    # (  896,  896,   32, True , False),
)

print "|     M|     N|     K| Op|OpenAI_32|cuBLAS_32|ratio_32|OpenAI_16|cuBLAS_16|ratio_16|"
print "|------|------|------|---|---------|---------|--------|---------|---------|--------|"

for m, n, k, at, bt in config:

    dimA = (k,m) if at else (m,k)
    dimB = (n,k) if bt else (k,n)
    dimC = (m,n)

    opA = 'T' if at else 'N'
    opB = 'T' if bt else 'N'
    op  = opA + opB

    dtype_data = list()

    for dtype in ( np.float32, np.float16 ): #np.float32, np.float16,

        A = ng.empty(dimA, dtype=dtype)
        B = ng.empty(dimB, dtype=dtype)
        C = ng.empty(dimC, dtype=dtype)

        if at: A = A.T
        if bt: B = B.T

        data = matmul(A, B, C, bench=True)

        # if dtype is np.float16:
        #     print ""
        #     for d in sorted(data):
        #         print "%7.3f %5.0f %22s %5d" % d

        cublas = data.pop()
        openai = sorted(data)[0]

        text = "%9.0f|%9.0f|%8.1f" % (openai[1], cublas[1], openai[1] / cublas[1])

        dtype_data.append(text)


    print "|%6d|%6d|%6d|%3s|%s|" % (m, n, k, op, "|".join(dtype_data))
