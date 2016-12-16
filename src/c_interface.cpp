#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <stdint.h>
#include <tuple>
#include <vector>
#include <unordered_map>


typedef struct CUfunc_st *CUfunction;
typedef struct CUmod_st *CUmodule;
typedef struct CUstream_st *CUstream;
typedef int CUdevice;

//include the cuda function prototypes here so that we don't need to know
//anything about cuda to compile this file - makes building with bazel
//a million times easier
extern "C" {

int
#ifdef _WIN32
__stdcall
#endif
cuModuleLoadData(CUmodule *, const void *);

int
#ifdef _WIN32
__stdcall
#endif
cuModuleGetFunction(CUfunction *, CUmodule, const char *);

int
#ifdef _WIN32
__stdcall
#endif
cuDeviceGetAttribute(int *, int, CUdevice);

int
#ifdef _WIN32
__stdcall
#endif
cuLaunchKernel(CUfunction,
               unsigned int,
               unsigned int,
               unsigned int,
               unsigned int,
               unsigned int,
               unsigned int,
               unsigned int,
               CUstream,
               void **,
               void **);

int
#ifdef _WIN32
__stdcall
#endif
cuCtxGetDevice(CUdevice *);

};

#include "include/c_interface.h"
#include "include/kernel_headers.h"

namespace {
#include "include/static_kernel_information.h"

std::mutex load_kernel_mutex_;

std::unordered_map<std::string, CUfunction> kernels_;

bool loadKernelsHelper(const std::unordered_map<std::string, const uint8_t*>& kernels) {
  for (auto kernel : kernels) {
    if (kernels_.count(kernel.first) > 0)
      continue;

    CUmodule module;

    int res = cuModuleLoadData(&module, kernel.second);
    if (res != 0) {
      std::cerr << "Failed to load " << kernel.first << " " <<
                res << std::endl;
      return false;
    }

    CUfunction function;

    std::string kernel_name = kernel.first.substr(0, kernel.first.size() - 6);

    res = cuModuleGetFunction(&function, module, kernel_name.c_str());
    if (res != 0) {
      std::cerr << "Failed to extract " << kernel_name << " " <<
                res << std::endl;
      return false;
    }

    kernels_.insert(std::make_pair(kernel.first, function));
  }

  return true;
}

bool loadKernels(int major) {
  std::lock_guard<std::mutex> lock(load_kernel_mutex_);

  if (major == 5)
    return loadKernelsHelper(kernels_50);
  else if (major == 6)
    return loadKernelsHelper(kernels_60);
  else {
    std::cerr << "Arch must be 5 or 6" << std::endl;
    return false;
  }
}

std::tuple<int, int, int> getDeviceProperties(CUdevice& device) {
  int major, minor;
  int res = cuDeviceGetAttribute(&major, 75, device);
  if (res != 0)
    return std::make_tuple(res, -1, -1);

  res = cuDeviceGetAttribute(&minor, 76, device);
  if (res != 0)
    return std::make_tuple(res, -1, -1);

  return std::make_tuple(0, major, minor);
}

std::pair<int, int> closest_divisor(int val, int div) {
  if (div == 2) {
    if ((val & 1) == 0) { return std::make_pair(2, val >> 1);  }
    else                { return std::make_pair(1, val);       }
  }
  else if (div == 4) {
    if      ((val & 3) == 0) { return std::make_pair(4, val >> 2); }
    else if ((val % 3) == 0) { return std::make_pair(3, val / 3);  }
    else if ((val % 5) == 0) { return std::make_pair(5, val / 5);  }
    else if ((val & 1) == 0) { return std::make_pair(2, val >> 1); }
    else if ((val % 7) == 0) { return std::make_pair(7, val / 7);  }
    else                     { return std::make_pair(1, val);      }
  }
}

std::string get_op_string(bool a_t, bool b_t) {
  if      (!a_t && !b_t) return "NN";
  else if (a_t && !b_t)  return "TN";
  else if (!a_t && b_t)  return "NT";
  else                   return "TT";
}

bool gemm(std::string precision, void *A, void *B, void *C,
          bool a_t, bool b_t,
          int m, int n, int k,
          int lda, int ldb, int ldc,
          float alpha, float beta,
          CUstream stream, unsigned int grid, unsigned int shared) {
  std::string kernel_op = get_op_string(a_t, b_t);

  if (grid >= selections[precision][kernel_op].size())
    return false;

  kernel_properties kp = selections[precision][kernel_op][grid];

  if (shared >= kp.shared_sizes.size())
    return false;

  bool vec4A, vec8A;
  bool vec4B, vec8B;
  if (a_t) {
    vec4A = (lda & 3) == 0 && (m & 3) == 0; //multiple of 4
    vec8A = (lda & 7) == 0 && (m & 7) == 0; //multiple of 8
  }
  else {
    vec4A = (lda & 3) == 0 && (k & 3) == 0; //multiple of 4
    vec8A = (lda & 7) == 0 && (k & 7) == 0; //multiple of 8
  }

  if (b_t) {
    vec4B = (ldb & 3) == 0 && (k & 3) == 0; //multiple of 4
    vec8B = (ldb & 7) == 0 && (k & 7) == 0; //multiple of 8
  }
  else {
    vec4B = (ldb & 3) == 0 && (n & 3) == 0; //multiple of 4
    vec8B = (ldb & 7) == 0 && (n & 7) == 0; //multiple of 8
  }

  bool vec4C = (ldc & 3) == 0 && (n & 3) == 0;

  bool vecA = (kp.vA == 4 && vec4A) || (kp.vA == 8 && vec8A);
  bool vecB = (kp.vB == 4 && vec4B) || (kp.vB == 8 && vec8B);
  bool vecC = kp.vC == 1 || vec4C;

  bool vec = vecA && vecB && vecC;

  CUdevice device;
  int res = cuCtxGetDevice(&device);
  if (res != 0)
    return false;

  bool success;
  int major, minor;

  std::tie(success, major, minor) = getDeviceProperties(device);

  if (success != 0)
    return false;

  std::string kernel_string;
  kernel_string.reserve(64);

  kernel_string += precision + "gemm_" + kp.tile_string +
                   "_" + kernel_op;
  if (vec)
    kernel_string += "_vec";

  if (major == 5)
    kernel_string += "_sm_50";
  else if (major == 6)
    kernel_string += "_sm_60";
  else
    return false;

  auto kernel = kernels_.find(kernel_string);
  if (kernel == kernels_.end()) {
    loadKernels(major);
    kernel = kernels_.find(kernel_string);
  }

  int blk_A = (m + kp.tile_m - 1) / kp.tile_m;
  int blk_B = (n + kp.tile_n - 1) / kp.tile_n;

  int blk_a, blk_b;
  std::tie(blk_a, blk_A) = closest_divisor(blk_A, kp.div);
  std::tie(blk_b, blk_B) = closest_divisor(blk_B, kp.div);

  if (blk_a == 1)
    std::tie(blk_a, blk_A) = std::make_pair(blk_A, 1);

  void *args[13] = {&C, &A, &B, &alpha, &beta, &lda, &ldb, &ldc,
                    &m, &n, &k, &blk_a, &blk_b};

  res = cuLaunchKernel(kernel->second, blk_a * blk_b, blk_B, blk_A,
                       kp.threads, 1, 1,
                       kp.shared_sizes[shared], stream, args, NULL);

  if (res != 0) {
    std::cerr << "Failed to execute " << kernel_string << " " <<
      res << std::endl;
    return false;
  }

  return true;
}

};

bool get_grid_limits(char precision, bool a_t, bool b_t, unsigned int *grid)
{
  std::string prec_string(1, precision);

  *grid = selections[prec_string][get_op_string(a_t, b_t)].size();
  return true;
}

bool get_shared_limits(char precision, bool a_t, bool b_t, unsigned int grid, unsigned int *shared) {
  std::string prec_string(1, precision);

  if (grid >= selections[prec_string][get_op_string(a_t, b_t)].size())
    return false;

  *shared = selections[prec_string][get_op_string(a_t, b_t)][grid].shared_sizes.size();

  return true;
}

bool openai_sgemm(float *A, float *B, float *C,
           bool a_t, bool b_t,
           int m, int n, int k,
           int lda, int ldb, int ldc,
           float alpha, float beta,
           CUstream stream, unsigned int grid, unsigned int shared) {
  return gemm("s",
              static_cast<void*>(A),
              static_cast<void*>(B),
              static_cast<void*>(C),
              a_t, b_t, m, n, k, lda, ldb, ldc,
              alpha, beta, stream, grid, shared);
}

bool openai_hgemm(uint16_t *A, uint16_t *B, uint16_t *C,
           bool a_t, bool b_t,
           int m, int n, int k,
           int lda, int ldb, int ldc,
           float alpha, float beta,
           CUstream stream, unsigned int grid, unsigned int shared) {
  return gemm("h",
              static_cast<void*>(A),
              static_cast<void*>(B),
              static_cast<void*>(C),
              a_t, b_t, m, n, k, lda, ldb, ldc,
              alpha, beta, stream, grid, shared);
}
