#include <sys/time.h>
#include <cassert>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_runtime_api.h>

using namespace nvcuda;


struct timeval tv0, tv1;

void begin_roi() {
  gettimeofday(&tv0, nullptr);
}

#define TV_TO_SEC(tv) (tv.tv_sec * 1000000 + tv.tv_usec)

void end_roi() {
  gettimeofday(&tv1, nullptr);
  std::cout << TV_TO_SEC(tv1) - TV_TO_SEC(tv0) << std::endl;
}

extern "C" __global__ void default_function_kernel0( half* __restrict__ a,  half* __restrict__ b,  float* __restrict__ c) {

  for (int x_outer_inner = 0; x_outer_inner < 4; ++x_outer_inner) {
    for (int y_outer_inner = 0; y_outer_inner < 4; ++y_outer_inner) {

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

      wmma::fill_fragment(c_frag, 0.0f);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;


      for (int rv_outer = 0; rv_outer < 256; ++rv_outer) {

        half *ptr_a = &a[((((((int)blockIdx.x) * 262144) + (x_outer_inner * 65536)) + (rv_outer * 16)))];
        wmma::load_matrix_sync(a_frag, ptr_a, 4096);
        half *ptr_b = &b[((((((int)threadIdx.x) * 262144) + (y_outer_inner * 65536)) + (rv_outer * 16)))];
        wmma::load_matrix_sync(b_frag, ptr_b, 4096);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

      }
      __syncthreads();

      float *ptr_c = &c[((((((((int)blockIdx.x) * 262144) + (x_outer_inner * 65536))) + (((int)threadIdx.x) * 64)) + (y_outer_inner * 16)))];
      wmma::store_matrix_sync(ptr_c, c_frag, 4096, wmma::mem_row_major);

    }
  }
}

int main() {
  half *a, *b;
  float *c;

  cudaMalloc(&a, 4096 * 4096 * (sizeof (half)));
  cudaMalloc(&b, 4096 * 4096 * (sizeof (half)));
  cudaMalloc(&c, 4096 * 4096 * (sizeof (float)));

  begin_roi();
  default_function_kernel0<<<64, 64>>>(a, b, c);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  end_roi();

  return 0;
}
