#pragma once

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

bool openai_sgemm(float *A, float *B, float *C,
           bool a_t, bool b_t,
           int m, int n, int k,
           int lda, int ldb, int ldc,
           float alpha, float beta,
           CUstream stream, int grid, int shared);

bool openai_hgemm(uint16_t *A, uint16_t *B, uint16_t *C,
           bool a_t, bool b_t,
           int m, int n, int k,
           int lda, int ldb, int ldc,
           float alpha, float beta,
           CUstream stream, int grid, int shared);

bool get_grid_limits(char precision, bool a_t, bool b_t, int *grid);
bool get_shared_limits(char precision, bool a_t, bool b_t, int grid, int *shared);

#ifdef __cplusplus
}
#endif
