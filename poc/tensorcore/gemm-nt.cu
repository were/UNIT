// TODO(@were): This is not correct yet!

#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#define N 128
#define M 128
#define K 128

using namespace nvcuda;

__global__ void foo(half *a, half *b, float *c) {
  int x = blockIdx.x / (M / 16);
  int y = blockIdx.x % (M / 16);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float, void> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, a + (x * 16) * K + k, K);
    wmma::load_matrix_sync(b_frag, b + (y * 16) * K + k, M);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(c + (x * 16) * K + (y * 16), c_frag, K, wmma::mem_row_major);
}

half a[N * K], b[M * K];
float c[N * M], ref[N * M];

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

  for (int i = 0; i < N * K; ++i)
    a[i] = __float2half((float)(rand() % 100) / 100.);
  for (int i = 0; i < M * K; ++i)
    b[i] = __float2half((float)(rand() % 100) / 100.);
  for (int i = 0; i < N * M; ++i)
    c[i] = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      ref[i * M + j] = 0.0;
      for (int k = 0; k < K; ++k)
        ref[i * M + j] += __half2float(a[i * K + k]) * __half2float(b[j * K + k]);
    }
  half *dev_a, *dev_b;
  float *dev_c;
  cudaMalloc(&dev_a, N * K * sizeof(half));
  cudaMalloc(&dev_b, M * K * sizeof(half));
  cudaMalloc(&dev_c, N * M * sizeof(float));
  cudaMemcpy(dev_a, a, sizeof a, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof b, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, sizeof c, cudaMemcpyHostToDevice);
  foo<<<N * M / 256, 32>>>(dev_a, dev_b, dev_c);
  cudaDeviceSynchronize();
  cudaMemcpy(c, dev_c, sizeof c, cudaMemcpyDeviceToHost);
  std::cout.precision(1);
  std::cout << std::fixed;
  for (int i = 0; i < N * M; ++i) {
    if (fabs(c[i] - ref[i]) > 1e-5) {
      std::cerr << "diff@" << i << std::endl;
      throw;
    }
  }
  //print(N, M, a);
  //print(N, M, b);
  //print(N, K, c);
  //print(N, M, ref);
  return 0;
}
