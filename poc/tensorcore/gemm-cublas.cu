#include <cassert>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "../util.h"

int m, n, k;

int main() {
  //std::cin >> m >> n >> k;
  m = 128;
  n = 768;
  k = 3072;
  cublasHandle_t handle;
  half alpha = __float2half(1.0f);
  half *a, *b;
  float *c;
  cudaMalloc(&a, m * k * sizeof(half));
  cudaMalloc(&b, n * k * sizeof(half));
  cudaMalloc(&c, n * m * sizeof(float));
  cublasCreate_v2(&handle);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  
  assert(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k, &alpha,
    a, CUDA_R_16F, m,
    b, CUDA_R_16F, k, &alpha,
    c, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  begin_roi();
  assert(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k, &alpha,
    a, CUDA_R_16F, m,
    b, CUDA_R_16F, k, &alpha,
    c, CUDA_R_32F, m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  float elps = end_roi();
  std::cout << (n * m * k) / elps / 1000 << std::endl;
  std::cout << elps << std::endl;
  return 0;
}
