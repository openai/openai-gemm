#!/usr/bin/env python

import numpy as np
import pycuda.driver as drv
from neon.backends.nervanagpu import NervanaGPU

from openai_gemm import matmul_test

ng = NervanaGPU()
print drv.Context.get_current().get_device().name()

ones = 0
out  = 0

# for i in range(1000): # np.float32, np.float16

#     matmul_test(ng, np.float32, "TN", 4096*4, 4096*4, 33, ones=ones, out=out) # update

#     if i % 100 == 0: print i

# exit()

small_1  = (1,2,3,4,5,6,7,8,9,16,32,64,65,72,120,127,128,192)
medium_1 = (32,64,128,192,778,785,786,787,794)
big_1    = (32,64,128,1532,1535,1536,1537,1540,3073,4095)

small_2  = (8,16,32,64,72,96,120,128,192)
medium_2 = (32,64,128,192,256,768-4,768-8,768,768+16,768+32)
big_2    = (32,64,128,1536-12,1536-24,1536,1536+28,1536+32,3072,4096)

for dtype in (np.float32, np.float16, ): # np.float32, np.float16
    print dtype

    for size in (small_1, small_2, medium_1, medium_2, big_1, big_2,): # small_1, small_2, medium_1, medium_2, big_1, big_2
        print size

        for K in size:
            print "K:", K

            for C in (size):
                print "C:", C

                for N in (size):

                    matmul_test(ng, dtype, "NN", N, K, C, ones=ones, out=out) # fprop
                    matmul_test(ng, dtype, "NT", N, C, K, ones=ones, out=out) # bprop
                    matmul_test(ng, dtype, "TN", C, K, N, ones=ones, out=out) # update
                    matmul_test(ng, dtype, "TT", K, N, C, ones=ones, out=out) # ------
