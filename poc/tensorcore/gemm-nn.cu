#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define N 32
#define M 32
#define K 32

using namespace nvcuda;

__global__ void foo(half *a, half *b, float *c) {
  int x = blockIdx.x / 2;
  int y = blockIdx.x % 2;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float, void> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < M; k += 16) {
    wmma::load_matrix_sync(a_frag, a + (x * 16) * M + k, M);
    wmma::load_matrix_sync(b_frag, b + K * k + y * 16, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c + (x * 16) * M + (y * 16), c_frag, K, wmma::mem_row_major);
}

half a[N * M], b[M * K];
float c[N * K], ref[N * K];

template<typename T>
void print(int n, int m, const T* a) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (j) std::cout << " ";
      std::cout << a[i * m + j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template<>
void print(int n, int m, const half* a) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (j) std::cout << " ";
      std::cout << __half2float(a[i * m + j]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  cudaDeviceProp prop;
  assert(cudaSuccess == cudaGetDeviceProperties(&prop, 0));
  std::cout << "Warp size is: " <<  prop.warpSize << std::endl;

  for (int i = 0; i < N * M; ++i)
    a[i] = __float2half((float)(rand() % 100) / 100.);
  for (int i = 0; i < M * K; ++i)
    b[i] = __float2half((float)(rand() % 100) / 100.);
  for (int i = 0; i < N * K; ++i)
    c[i] = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < K; ++j) {
      ref[i * K + j] = 0.0;
      for (int k = 0; k < M; ++k)
        ref[i * K + j] += __half2float(a[i * M + k]) * __half2float(b[k * K + j]);
    }
  half *dev_a, *dev_b;
  float *dev_c;
  cudaMalloc(&dev_a, N * M * sizeof(half));
  cudaMalloc(&dev_b, M * K * sizeof(half));
  cudaMalloc(&dev_c, N * K * sizeof(float));
  cudaMemcpy(dev_a, a, sizeof a, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof b, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, sizeof c, cudaMemcpyHostToDevice);
  foo<<<4, 32>>>(dev_a, dev_b, dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(c, dev_c, sizeof c, cudaMemcpyDeviceToHost);
  std::cout.precision(1);
  std::cout << std::fixed;
  for (int i = 0; i < N * M; ++i)
    assert(fabs(c[i] - ref[i]) < 1e-5);
  //print(N, M, a);
  //print(N, M, b);
  //print(N, K, c);
  //print(N, M, ref);
  return 0;
}
