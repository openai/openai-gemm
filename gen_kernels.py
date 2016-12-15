import glob
import os.path
import re
import subprocess
import sys
import time

base_dir  = os.path.dirname(__file__)
maxas_dir = os.path.join(base_dir, "maxas")
sass_dir  = os.path.join(base_dir, "sass")

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


def _get_cache_dir(subdir=None):
    cache_dir = 'temp/'

    if subdir:
        subdir = subdir if isinstance(subdir, list) else [subdir]
        cache_dir = os.path.join(cache_dir, *subdir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir


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


def run_command(cmdlist):
    cmd  = " ".join(cmdlist)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("Error(%d):\n%s\n%s" % (proc.returncode, cmd, err))


def get_kernel(base_name, major, minor, options=None):
    if major < 5:
        raise RuntimeError("sass kernels require Maxwell or greater class hardware")
    elif major >= 7:
        raise RuntimeError("sm version 7 or greater is not supported")

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
    header_dir = os.path.join(base_dir, "include/kernels")

    ptx_version = "4.2" if major < 6 else "5.0"
    ptx_file   = get_ptx_file(kernel_spec, kernel_name, arch, ptx_version)
    cubin_file = os.path.join(cubin_dir, cubin_name)
    sass_file   = os.path.join(sass_dir, sass_name)
    header_file = os.path.join(header_dir, kernel_name + "_" + arch + ".h")

    if not os.path.exists(sass_file):
        raise RuntimeError("Missing: %s for kernel: %s" % (sass_name, kernel_name))

    # build the cubin and run maxas in the same command
    # we don't want the chance of a generated cubin not processed by maxas (in case user hits ^C in between these steps)
    command_string = [ "ptxas -v -arch", arch, "-o", cubin_file, ptx_file, ";" ] + maxas_i + [sass_file, cubin_file]
    run_command(command_string)
    cubin_mtime = time.time()

    # now also generate the associated header file containing the cubin
    with open(cubin_file, 'rb') as input_file:
        with open(header_file, 'wb') as output_file:
            output_file.write('const uint8_t %s[] = {' % (kernel_name + "_" + arch))
            byte = input_file.read(1)
            count = 0
            while byte:
                if count % 12 == 0:
                    output_file.write('\n   ')
                output_file.write(' 0x' + byte.encode('hex') + ',')
                byte = input_file.read(1)
                count += 1
            output_file.write('\n};')


def gen_kernels():
    for prefix in ['s', 'h']:
        for op in ['NN', 'NT', 'TN', 'TT']:
            for tileM, tileN, tileK, vecA, vecB, vecC, div, base_op, dyn_shared in selections[prefix][op]:
                for vec in [False, True]:
                    for major, minor in [(5, 0), (6, 0)]:
                        if base_op:
                            # The op is part of the base kernel name
                            base = "%sgemm_%dx%dx%d_%s" % (prefix, tileM, tileN, tileK, op)
                            opts = ( "vec", ) if vec else ()
                        else:
                            # The op is an option passed to a more generic kernel
                            base = "%sgemm_%dx%dx%d" % (prefix, tileM, tileN, tileK)
                            opts = ( op, "vec" ) if vec else (op,)

                        get_kernel(base, major, minor, opts)


def main():
    gen_kernels()


if __name__ == "__main__":
    main()
