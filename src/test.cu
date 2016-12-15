#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <vector>

#include "include/c_interface.h"

/* Simple program to call all possible kernel variants to make sure the are
   callable and produce sane results.  Not meant to check the correctness of
   the underlying routines */

int main(void) {
  cudaFree(0);

  std::vector<std::pair<bool, bool>> ops = { {false, false}, {false, true}, 
                                             {true, false}, {true, true} };
  {
    float *A, *B, *C;
    const int size = 1024;

    cudaMalloc(&A, size * size * sizeof(float));
    cudaMalloc(&B, size * size * sizeof(float));
    cudaMalloc(&C, size * size * sizeof(float));
    float *C_host = (float *)malloc(size * size * sizeof(float));

    thrust::fill_n(thrust::device, A, size * size, 1.f);
    thrust::fill_n(thrust::device, B, size * size, 1.f);

    for (auto op : ops) {
      unsigned int grid;
      get_grid_limits('s', op.first, op.second, &grid);
      for (int g = 0; g < grid; ++g) {
        unsigned int shared;
        get_shared_limits('s', op.first, op.second, g, &shared);
        for (int s = 0; s < shared; ++s) {
          bool res = openai_sgemm(A, B, C, op.first, op.second, size, size, size,
                       size, size, size, 1.0, 0.0, NULL, g, s);
          assert(res);
          cudaMemcpy(C_host, C, size * size * sizeof(float), cudaMemcpyDeviceToHost);
          
          assert(C_host[0] == 1024);
        }
      }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(C_host);
  }

  {
    uint16_t *A, *B, *C;
    const int size = 1024;

    cudaMalloc(&A, size * size * sizeof(uint16_t));
    cudaMalloc(&B, size * size * sizeof(uint16_t));
    cudaMalloc(&C, size * size * sizeof(uint16_t));
    uint16_t *C_host = (uint16_t *)malloc(size * size * sizeof(uint16_t));

    thrust::fill_n(thrust::device, A, size * size, 0x3c00);
    thrust::fill_n(thrust::device, B, size * size, 0x3c00);

    for (auto op : ops) {
      unsigned int grid;
      get_grid_limits('h', op.first, op.second, &grid);
      for (int g = 0; g < grid; ++g) {
        unsigned int shared;
        get_shared_limits('h', op.first, op.second, g, &shared);
        for (int s = 0; s < shared; ++s) {
          bool res = openai_hgemm(A, B, C, op.first, op.second, size, size, size,
                       size, size, size, 1.0, 0.0, NULL, g, s);
          assert(res);
          cudaMemcpy(C_host, C, size * size * sizeof(uint16_t), cudaMemcpyDeviceToHost);

          assert(C_host[0] == 25600);
        }
      }
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(C_host);
  }

  return 0;
}
