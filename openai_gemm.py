import re
import time
import ctypes
import appdirs
import os.path
import subprocess
import numpy as np
import pycuda.driver as drv
from operator import mul
from pycuda.tools import context_dependent_memoize
from scikits.cuda import cublas

def matmul(A, B, C, alpha=1.0, beta=0.0, stream=None, bench=False):
    """
        C = alpha * A   . B   + beta * C
        C = alpha * A.T . B   + beta * C
        C = alpha * A   . B.T + beta * C
        C = alpha * A.T . B.T + beta * C

        bench: return benchmark data for all available tiles + cublas
    """

    # this could be relaxed, kernels are capable of mixed precision (with minor tweaks)
    # the s/h prefix would then go away and each type would be specified with kernel build option
    assert A.dtype.type == B.dtype.type == C.dtype.type

    if   C.dtype.type is np.float32:
        prefix = "s"
    elif C.dtype.type is np.float16:
        prefix = "h"
    else:
        raise TypeError("Only floating point dot currently supported.")

    # (m,n) = (m,k) . (k,n)
    m = A.shape[0]
    n = B.shape[1]
    k = A.shape[1]
    assert m == C.shape[0]
    assert n == C.shape[1]
    assert k == B.shape[0]

    # Extract the operations and contiguous dimension sizes (cda, cdb, cdc).
    # Note that these can be the same as from the shape unless the non-contiguous dimension is sliced.
    # One dimension must be contiguous (DRAM efficiency demands this).
    # Note that the strides here do not include the datatype size as they would in numpy.
    # A transpose op (.T) on a GPUTensor reverses the shape and strides then flags the tensor as transposed (is_trans=True) -
    #    The underlying data is unchanged.
    if A.is_trans:
         opA  = 'T'
         cda  = A.strides[1]
         assert A.strides[0] == 1
    else:
         opA  = 'N'
         cda  = A.strides[0]
         assert A.strides[1] == 1

    if B.is_trans:
         opB  = 'T'
         cdb  = B.strides[1]
         assert B.strides[0] == 1
    else:
         opB  = 'N'
         cdb  = B.strides[0]
         assert B.strides[1] == 1

    cdc  = C.strides[0]
    assert C.strides[1] == 1

    op = opA + opB

    # get and autotune the kernel selection
    kernel, params, dynamic_shared = _get_gemm_kernel(prefix, op, cda, cdb, cdc, m, n, k)

    # bind dynamic params
    params[2:8] = (stream, C.gpudata, A.gpudata, B.gpudata, alpha, beta)

    # call the kernel
    kernel.prepared_async_call(*params, shared_size=dynamic_shared)

    # unbind dynamic params
    params[2:8] = (None,) * 6

    # return benchmark data if requested
    if bench:
        return _get_bench_data()[(prefix, op, cda, cdb, cdc, m, n, k)]

    return C



####################################################################################################


# scikits.cuda doesn't expose cublasSgemmEx
cublas._libcublas.cublasSgemmEx.restype  = int
cublas._libcublas.cublasSgemmEx.argtypes = [
    cublas._types.handle,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int ]

def cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    status = cublas._libcublas.cublasSgemmEx(handle,
        cublas._CUBLAS_OP[transa], cublas._CUBLAS_OP[transb], m, n, k,
        ctypes.byref(ctypes.c_float(alpha)),
        int(A), 2, lda,
        int(B), 2, ldb,
        ctypes.byref(ctypes.c_float(beta)),
        int(C), 2, ldc)
    cublas.cublasCheckStatus(status)

cublasXgemm = {
    "s" : cublas.cublasSgemm,
    "h" : cublasHgemm,
}


@context_dependent_memoize
def _get_sm_count():
    attributes = drv.Context.get_device().get_attributes()
    return attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]

@context_dependent_memoize
def _get_events():
    return (drv.Event(), drv.Event())

@context_dependent_memoize
def _get_cublas():
    return cublas.cublasCreate()

@context_dependent_memoize
def _get_bench_data():
    return dict()

def _ceil_div(x, y):
    return -(-x // y)

def _closest_divisor(val, div):
    divisors = sorted([(abs(i - div), i) for i in range(2, 8) if val % i == 0])
    if len(divisors):
        return (divisors[0][1], val // divisors[0][1])
    else:
        return (1, val)


# Tile sizes:      m,   n,  k, vA,vB,vC  div,  op (dynamic shared options)
k128x128x8    = (128, 128,  8,  4, 4, 1,   2,  0, (0,))
k32x32x32     = ( 32,  32, 32,  4, 4, 1,   4,  0, (0, 2**14))
k32x64x32_NN  = ( 32,  64, 32,  8, 4, 4,   4,  1, (0, 2**13))
k32x32x64_NT  = ( 32,  32, 64,  8, 8, 4,   4,  1, (0,))
k16x64x64_NN  = ( 16,  64, 64,  8, 4, 4,   4,  1, (0,))
k16x64x64_NT  = ( 16,  64, 64,  8, 8, 4,   4,  1, (0,))

selections = {
    "s" : {
        "TN" : (k128x128x8, k32x32x32),
        "NN" : (k128x128x8, k32x32x32),
        "NT" : (k128x128x8, k32x32x32),
        "TT" : (k128x128x8, k32x32x32),
    },
    "h" : {
        "TN" : (k128x128x8, k32x32x32),
        "NN" : (k128x128x8, k32x32x32, k32x64x32_NN, k16x64x64_NN),
        "NT" : (k128x128x8, k32x32x32, k32x32x64_NT, k16x64x64_NT),
        "TT" : (k128x128x8, k32x32x32),
    },
}

# Autotune kernel selection
@context_dependent_memoize
def _get_gemm_kernel(prefix, op, cda, cdb, cdc, m, n, k):

    if op[0] == 'T':
         vec4A = (cda & 3) == 0 and (m & 3) == 0
         vec8A = (cda & 7) == 0 and (m & 7) == 0
         dimA  = (k,cda)
    else:
         vec4A = (cda & 3) == 0 and (k & 3) == 0
         vec8A = (cda & 7) == 0 and (k & 7) == 0
         dimA  = (m,cda)

    if op[1] == 'T':
         vec4B = (cdb & 3) == 0 and (k & 3) == 0
         vec8B = (cdb & 7) == 0 and (k & 7) == 0
         dimB  = (n,cdb)
    else:
         vec4B = (cdb & 3) == 0 and (n & 3) == 0
         vec8B = (cdb & 7) == 0 and (n & 7) == 0
         dimB  = (k,cdb)

    vec4C = (cdc & 3) == 0 and (n & 3) == 0
    dimC  = (m,cdc)

    dtype = np.dtype(np.float32 if prefix == 's' else np.float16)

    A = drv.mem_alloc(mul(*dimA) * dtype.itemsize)
    B = drv.mem_alloc(mul(*dimB) * dtype.itemsize)
    C = drv.mem_alloc(mul(*dimC) * dtype.itemsize)

    # TODO: use curand
    dataA = np.random.uniform(-1.0, 1.0, dimA).astype(dtype)
    dataB = np.random.uniform(-1.0, 1.0, dimB).astype(dtype)
    drv.memcpy_htod(int(A), dataA)
    drv.memcpy_htod(int(B), dataB)

    # Using random data gets you more accurate autotune results
    # drv.memset_d8(int(A), 0, mul(*dimA) * dtype.itemsize)
    # drv.memset_d8(int(B), 0, mul(*dimB) * dtype.itemsize)

    timings = []
    cache   = []

    # scale the repeat count to amount of work
    repeat = min(max(int(5e11 * 28 / (m*n*k * 2.0 * _get_sm_count()) ), 10), 5000)
    warmup = repeat
    #print repeat

    start, end = _get_events()
    flops = m * n * k * 2.0

    for tileM, tileN, tileK, vecA, vecB, vecC, div, base_op, dyn_shared in selections[prefix][op]:

        vecA = (vecA == 4 and vec4A) or (vecA == 8 and vec8A)
        vecB = (vecB == 4 and vec4B) or (vecB == 8 and vec8B)
        vecC =  vecC == 1 or  vec4C
        vec  = vecA and vecB and vecC

        if base_op:
            # The op is part of the base kernel name
            base = "%sgemm_%dx%dx%d_%s" % (prefix, tileM, tileN, tileK, op)
            opts = ( "vec", ) if vec else ()
        else:
            # The op is an option passed to a more generic kernel
            base = "%sgemm_%dx%dx%d" % (prefix, tileM, tileN, tileK)
            opts = ( op, "vec" ) if vec else (op,)

        kernel = get_kernel(base, opts)

        blk_A = _ceil_div(m, tileM)
        blk_B = _ceil_div(n, tileN)

        # TODO: perhaps autotune all possible small divisors
        blk_a, blk_A = _closest_divisor(blk_A, div)
        blk_b, blk_B = _closest_divisor(blk_B, div)
        if blk_a == 1:
            blk_a, blk_A = (blk_A, 1)

        for dynamic_shared in dyn_shared:

            params = [
                (blk_a * blk_b, blk_B, blk_A), (kernel.threads, 1, 1), None,
                C, A, B, 1.0, 0.0,
                cda, cdb, cdc, m, n, k, blk_a, blk_b ]

            #print kernel.name, params, dynamic_shared

            # Warmup (once per config)
            for r in range(warmup):
                kernel.prepared_async_call(*params)
            warmup = 0

            # Benchmark
            start.record()
            for r in range(repeat):
                kernel.prepared_async_call(*params, shared_size=dynamic_shared)
            end.record()
            end.synchronize()
            msecs = end.time_since(start) / float(repeat)
            gflops = flops / (msecs * 1000000.0)

            params[3:8] = (None,) * 5

            timings.append((msecs, gflops, kernel, params, dynamic_shared))
            cache.append((msecs, gflops, kernel.name, dynamic_shared))

    # record a cublas time for reference
    cublas_handle = _get_cublas()
    start.record()
    for r in range(repeat):
        # convert row order to col order
        cublasXgemm[prefix](cublas_handle, op[1], op[0], n, m, k, 1.0, B, cdb, A, cda, 0.0, C, cdc)
    end.record()
    end.synchronize()
    msecs = end.time_since(start) / float(repeat)
    gflops = flops / (msecs * 1000000.0)
    cache.append( (msecs, gflops, "cuBLAS", 0) )

    # cache complete timing data for benchmark comparisons
    # this data could be cached to disk for quicker autotuning on future runs
    _get_bench_data()[(prefix, op, cda, cdb, cdc, m, n, k)] = cache

    # return the fastest kernel
    return tuple(sorted(timings)[0][2:5])



# Utility function to test all tiles for the given dimensions and dtype
def matmul_test(ng, dtype, op, m, n, k, ones=False, out=False):

    prefix = "s" if dtype is np.float32 else "h"

    if op[0] == 'T':
         vec4A = (m & 3) == 0
         vec8A = (m & 7) == 0
         dimA  = (k,m)
         cda   = m
    else:
         vec4A = (k & 3) == 0
         vec8A = (k & 7) == 0
         dimA  = (m,k)
         cda   = k

    if op[1] == 'T':
         vec4B = (k & 3) == 0
         vec8B = (k & 7) == 0
         dimB  = (n,k)
         cdb   = k
    else:
         vec4B = (n & 3) == 0
         vec8B = (n & 7) == 0
         dimB  = (k,n)
         cdb   = n

    vec4C = (n & 3) == 0
    dimC  = (m,n)
    cdc   = n

    A1 = ng.empty(dimA, dtype=dtype)
    B1 = ng.empty(dimB, dtype=dtype)
    C1 = ng.empty(dimC, dtype=dtype)
    C2 = ng.empty(dimC, dtype=dtype)

    if ones:
        A1[:] = 1.0
        B1[:] = 1.0
    else:
        # fill with uniform randoms from -1 to 1
        A1[:] = 2 * (.5 - ng.rand())
        B1[:] = 2 * (.5 - ng.rand())

    # for reducing outputs
    partial1 = ng.empty((C1.shape[0],1), dtype=np.float32)
    partial2 = partial1[0:1,0:1]

    cublas_handle = _get_cublas()

    for tileM, tileN, tileK, vecA, vecB, vecC, div, base_op, dyn_shared in selections[prefix][op]:

        vecA = (vecA == 4 and vec4A) or (vecA == 8 and vec8A)
        vecB = (vecB == 4 and vec4B) or (vecB == 8 and vec8B)
        vecC =  vecC == 1 or  vec4C
        vec  = vecA and vecB and vecC

        if base_op:
            # The op is part of the base kernel name
            base = "%sgemm_%dx%dx%d_%s" % (prefix, tileM, tileN, tileK, op)
            opts = ( "vec", ) if vec else ()
        else:
            # The op is an option passed to a more generic kernel
            base = "%sgemm_%dx%dx%d" % (prefix, tileM, tileN, tileK)
            opts = ( op, "vec" ) if vec else (op,)

        kernel = get_kernel(base, opts)

        blk_A = _ceil_div(m, tileM)
        blk_B = _ceil_div(n, tileN)

        blk_a, blk_A = _closest_divisor(blk_A, div)
        blk_b, blk_B = _closest_divisor(blk_B, div)
        if blk_a == 1:
            blk_a, blk_A = (blk_A, 1)

        for alpha, beta in ( (1.0,0.0), (0.5,0.5) ):

            try:
                if ones:
                    C1[:] = 1.0
                else:
                    C1[:] = 2 * (.5 - ng.rand())
                C2[:] = C1

                params = [
                    (blk_a * blk_b, blk_B, blk_A), (kernel.threads, 1, 1), None,
                    C1.gpudata, A1.gpudata, B1.gpudata, alpha, beta,
                    cda, cdb, cdc, m, n, k, blk_a, blk_b ]

                kernel.prepared_async_call(*params)

                # convert row order to col order
                cublasXgemm[prefix](cublas_handle, op[1], op[0], n, m, k, alpha, B1.gpudata, cdb, A1.gpudata, cda, beta, C2.gpudata, cdc)

                # Check for NaNs
                partial1[:] = ng.min(ng.finite(C1), axis=1)
                partial2[:] = ng.min(partial1, axis=0)
                if partial2.get()[0,0] == 0.0:
                    print "Error: NaN kernel: %s mnk: (%d,%d,%d) ab: (%f,%f)" % (kernel.name, m,n,k, alpha,beta)
                    exit()

                # Get Max Diff
                partial1[:] = ng.max(abs(C2 - C1), axis=1)
                partial2[:] = ng.max(partial1, axis=0)
                diff = partial2.get()[0,0]

                # Get Mean
                partial1[:] = ng.sum(abs(C2), axis=1)
                partial2[:] = ng.sum(partial1, axis=0)
                mean = partial2.get()[0,0] / C2.size

                # Scale diff by the mean
                pctErr = 100 * diff / mean

                #print "Error: %.3f %s" % (pctErr, kernel.name)

                maxerr = .005 if dtype is np.float32 else 0.7

                if pctErr > maxerr:
                    print "Error: %.3f%% diff: %.5f mean %.5f kernel: %s mnk: (%d,%d,%d) ab: (%f,%f)" % (pctErr, diff, mean, kernel.name, m,n,k, alpha,beta)
                    print params
                    if out:
                        C1  = C1.get()
                        C2  = C2.get()
                        D  = abs(C2 - C1)
                        np.savetxt("out_diff.txt",    D,  fmt='%3.1f')
                        np.savetxt("out_correct.txt", C2, fmt='%5.1f')
                        np.savetxt("out_error",       C1, fmt='%5.1f')
                    exit()

            except drv.Error as e:
                print "kernel: %s mnk: (%d,%d,%d) ab: (%f,%f)" % (kernel.name, m,n,k, alpha,beta)
                print e
                exit()

### below code adapted from Nervana Neon: kernel_specs.py

def _get_cache_dir(subdir=None):

    cache_dir = appdirs.user_cache_dir("openai-gemm")

    if subdir:
        subdir = subdir if isinstance(subdir, list) else [subdir]
        cache_dir = os.path.join(cache_dir, *subdir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir

# helpful for kernel development
debug = 0

base_dir  = os.path.dirname(__file__)
maxas_dir = os.path.join(base_dir, "maxas")
sass_dir  = os.path.join(base_dir, "sass")

kernels = {
    # Generic gemm tiles
    "sgemm_128x128x8":   {"threads": 256, "sass": "xgemm_128x128x8",   "params": "xgemm", "share": "(128*8 + 32)*4 + 4", "args": {"type": "s"} },
    "hgemm_128x128x8":   {"threads": 256, "sass": "xgemm_128x128x8",   "params": "xgemm", "share": "(128*8 + 32)*4 + 4", "args": {"type": "h"} },
    "sgemm_32x32x32":    {"threads": 128, "sass": "xgemm_32x32x32",    "params": "xgemm", "share": "(32*33)*4 + 4",      "args": {"type": "s"} },
    "hgemm_32x32x32":    {"threads": 128, "sass": "xgemm_32x32x32",    "params": "xgemm", "share": "(32*33)*4 + 4",      "args": {"type": "h"} },

    # Custom hgemm tiles designed for small minibatch RNNs
    "hgemm_32x64x32_NN": {"threads": 128, "sass": "hgemm_32x64x32_NN", "params": "xgemm", "share": "32*33*2 + 64*32*2 + 4"               },
    "hgemm_32x32x64_NT": {"threads": 128, "sass": "hgemm_32x32x64_NT", "params": "xgemm", "share": "32*65*4 + 4"                         },
    "hgemm_16x64x64_NN": {"threads": 128, "sass": "hgemm_16x64x64_NN", "params": "xgemm", "share": "(16*64 + 32)*2 + 64*64*2 + 4"        },
    "hgemm_16x64x64_NT": {"threads": 128, "sass": "hgemm_16x64x64_NT", "params": "xgemm", "share": "(16*64 + 32)*2 + (64*64 + 32)*2 + 4" },
}

_params = {
    "xgemm": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float param_alpha",
        "float param_beta",
        "unsigned param_cda",
        "unsigned param_cdb",
        "unsigned param_cdc",
        "unsigned param_m",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_blk_a",
        "unsigned param_blk_b",
    ],
}

_space_re = re.compile(r"\s+")

_share_template = r"""
    .shared .align 4 .b32 share[{0}];
"""

_kernel_template = r"""
.version {6}
.target {0}
.address_size 64

// args: {5}

.visible .entry  {1}(
{2}
)
.reqntid {3}
{{
{4}
    ret;
}}
"""

def get_ptx_file(kernel_spec, kernel_name, arch, ptx_ver):

    ptx_dir = _get_cache_dir([arch, 'ptx'])

    thread_spec = kernel_spec["threads"]
    args_spec   = str(kernel_spec.get("args",""))
    param_spec  = _params[kernel_spec["params"]]

    kernel_params = []
    for p in param_spec:
        ptype, pname = _space_re.split(p)

        if ptype[-1] == '*':
            ptype = '.u64'
        elif ptype == 'float':
            ptype = '.f32'
        else:
            ptype = '.u32'

        kernel_params.append("    .param %s %s" % (ptype, pname))

    kernel_params = ",\n".join(kernel_params)

    if "share" in kernel_spec:
        share = _share_template.format(eval(kernel_spec["share"]))
    else:
        share = ""

    kernel_text = _kernel_template.format(arch, kernel_name, kernel_params, thread_spec, share, args_spec, ptx_ver)
    kernel_ptx  = os.path.join(ptx_dir, kernel_name + ".ptx")

    current_text = ""
    if os.path.exists(kernel_ptx):
        f = open(kernel_ptx, "r")
        current_text = f.read()
        f.close()
    # only write out the kernel if text has changed.
    if kernel_text != current_text:
        f = open(kernel_ptx, "w")
        f.write(kernel_text)
        f.close()

    return kernel_ptx


include_re = re.compile(r'^<INCLUDE\s+file="([^"]+)"\s*/>')

def extract_includes(name, includes=None):
    if not includes:
        includes = list()
    sass_file = os.path.join(sass_dir, name)
    includes.append((sass_file, os.path.getmtime(sass_file)))
    for line in open(sass_file, "r"):
        match = include_re.search(line)
        if match:
            extract_includes(match.group(1), includes)
    return includes

def run_command(cmdlist):
    cmd  = " ".join(cmdlist)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("Error(%d):\n%s\n%s" % (proc.returncode, cmd, err))
    if debug:
        print cmd
        if out: print out
        if err: print err

@context_dependent_memoize
def get_kernel(base_name, options=None):

    attributes = drv.Context.get_device().get_attributes()
    major = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    minor = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]
    if major < 5:
        raise RuntimeError("sass kernels require Maxwell or greater class hardware")

    arch = "sm_%d%d" % (major, minor)

    libprefix = "PERL5LIB=%s" % maxas_dir
    maxas_i = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -i -w"]
    maxas_p = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -p"]

    kernel_spec = kernels[base_name]
    kernel_name = base_name

    # static options
    if "args" in kernel_spec:
        for pair in kernel_spec["args"].items():
            maxas_i.append("-D%s %s" % pair)
            maxas_p.append("-D%s %s" % pair)

    # dynamic options
    if options is not None:
        for opt in options:
            if type(opt) is tuple:
                maxas_i.append("-D%s %s" % opt)
                maxas_p.append("-D%s %s" % opt)
                kernel_name += "_%s%s" % opt
            else:
                maxas_i.append("-D%s 1" % opt)
                maxas_p.append("-D%s 1" % opt)
                kernel_name += "_%s" % opt

    maxas_i.insert(2, "-k " + kernel_name)

    sass_name  = kernel_spec["sass"] + ".sass"
    cubin_name = kernel_name + ".cubin"
    cubin_dir  = _get_cache_dir([arch, 'cubin'])

    ptx_version = "4.2" if major < 6 else "5.0"
    ptx_file   = get_ptx_file(kernel_spec, kernel_name, arch, ptx_version)
    sass_file  = os.path.join(sass_dir, sass_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)

    if not os.path.exists(sass_file):
        raise RuntimeError("Missing: %s for kernel: %s" % (sass_name, kernel_name))

    ptx_mtime   = os.path.getmtime(ptx_file)
    cubin_mtime = os.path.getmtime(cubin_file) if os.path.exists(cubin_file) else 0

    build_cubin = False
    if ptx_mtime > cubin_mtime:
        build_cubin = True

    includes = extract_includes(sass_name)
    for include, include_mtime in includes:
        if include_mtime > cubin_mtime:
            build_cubin = True
            break

    if build_cubin:
        # build the cubin and run maxas in the same command
        # we don't want the chance of a generated cubin not processed by maxas (in case user hits ^C in between these steps)
        run_command([ "ptxas -v -arch", arch, "-o", cubin_file, ptx_file, ";" ] + maxas_i + [sass_file, cubin_file])
        cubin_mtime = time.time()

    # output preprocessed and disassembled versions in debug mode
    if debug:
        pre_dir  = _get_cache_dir([arch, 'pre'])
        dump_dir = _get_cache_dir([arch, 'dump'])

        pre_file   = os.path.join(pre_dir,  kernel_name + "_pre.sass")
        dump_file  = os.path.join(dump_dir, kernel_name + "_dump.sass")
        pre_mtime  = os.path.getmtime(pre_file)  if os.path.exists(pre_file)  else 0
        dump_mtime = os.path.getmtime(dump_file) if os.path.exists(dump_file) else 0

        for include, include_mtime in includes:
            if include_mtime > pre_mtime:
                run_command(maxas_p + [sass_file, pre_file])
                break

        if cubin_mtime > dump_mtime:
            run_command(["nvdisasm -c", cubin_file, ">", dump_file])

    # generate the function signature for pycuda
    params  = _params[kernel_spec["params"]]
    sig = ""
    for p in params:
        ptype, pname = _space_re.split(p)
        if ptype[-1] == '*':
            sig += "Q"
        elif ptype == 'float':
            sig += "f"
        elif ptype == 'unsigned':
            sig += "I"
        else:
            sig += "i"

    module = drv.module_from_file(cubin_file)
    func   = module.get_function(kernel_name)
    func.prepare(sig)
    func.threads = kernel_spec["threads"]
    func.name = kernel_name
    func.static_shared = eval(kernel_spec["share"])

    return func