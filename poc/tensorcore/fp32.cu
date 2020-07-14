
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define N 16
#define M 16
#define K 16

using namespace nvcuda;

__global__ void foo(float *a, float *b, float *c) {
   wmma::fragment<wmma::matrix_a, N, M, K, float, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, N, M, K, float, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, N, M, K, float> c_frag;

   wmma::fill_fragment(c_frag, 2.0f);

   wmma::load_matrix_sync(a_frag, a, M);
   wmma::load_matrix_sync(b_frag, b, K);
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   wmma::store_matrix_sync(c, c_frag, K, wmma::mem_row_major);
}

float a[N * M], b[M * K];
float c[N * K], ref[N * K];

int main() {
  cudaDeviceProp prop;
  assert(cudaSuccess == cudaGetDeviceProperties(&prop, 0));
  std::cout << "Warp size is: " <<  prop.warpSize << std::endl;

  for (int i = 0; i < N * M; ++i)
    a[i] = rand();
  for (int i = 0; i < M * K; ++i)
    b[i] = rand();
  for (int i = 0; i < N * K; ++i)
    c[i] = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < K; ++j) {
      ref[i * K + j] = 0.0;
      for (int k = 0; k < M; ++k)
        ref[i * K + j] += __half2float(a[i * M + k]) * __half2float(b[k * K + j]);
    }
  float *dev_a, *dev_b;
  float *dev_c;
  cudaMalloc(&dev_a, N * M * sizeof(float));
  cudaMalloc(&dev_b, M * K * sizeof(float));
  cudaMalloc(&dev_c, N * K * sizeof(float));
  cudaMemcpy(dev_a, a, sizeof a, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof b, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, sizeof c, cudaMemcpyHostToDevice);
  foo<<<1, 1>>>(dev_a, dev_b, dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(c, dev_c, sizeof c, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      if (j) std::cout << " ";
      std::cout << c[i * K + j];// << "(" << ref[i * K + j] << ")";
    }
    std::cout << std::endl;
  }
  return 0;
}
