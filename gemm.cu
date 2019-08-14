#include <cuda_fp16.h>
extern "C" __global__ void default_function_kernel0( half* __restrict__ b,  half* __restrict__ a,  float* __restrict__ c) {
   float c_local[256];
  __shared__ half b_shared[4033];
  for (int x_outer_inner = 0; x_outer_inner < 4; ++x_outer_inner) {
    for (int y_outer_inner = 0; y_outer_inner < 4; ++y_outer_inner) {
      for (int x_c_init = 0; x_c_init < 16; ++x_c_init) {
        for (int y_c_init = 0; y_c_init < 16; ++y_c_init) {
          c_local[((x_c_init * 16) + y_c_init)] = 0.000000e+00f;
        }
      }
      for (int rv_outer = 0; rv_outer < 256; ++rv_outer) {
        for (int x_c = 0; x_c < 16; ++x_c) {
          for (int y_c = 0; y_c < 16; ++y_c) {
            for (int rv_inner = 0; rv_inner < 16; ++rv_inner) {
              __syncthreads();
              for (int ax0 = 0; ax0 < 4033; ++ax0) {
                b_shared[ax0] = b[(((((y_outer_inner * 65536) + (ax0 * 4096)) + (y_c * 4096)) + (rv_outer * 16)) + rv_inner)];
              }
              __syncthreads();
              c_local[((x_c * 16) + y_c)] = (c_local[((x_c * 16) + y_c)] + (((float)a[(((((((int)blockIdx.x) * 262144) + (x_outer_inner * 65536)) + (x_c * 4096)) + (rv_outer * 16)) + rv_inner)]) * ((float)b_shared[(((int)threadIdx.x) * 64)])));
            }
          }
        }
      }
      for (int x_inner = 0; x_inner < 16; ++x_inner) {
        for (int y_inner = 0; y_inner < 16; ++y_inner) {
          c[((((((((int)blockIdx.x) * 262144) + (x_outer_inner * 65536)) + (x_inner * 4096)) + (((int)threadIdx.x) * 64)) + (y_outer_inner * 16)) + y_inner)] = c_local[((x_inner * 16) + y_inner)];
        }
      }
    }
  }
}

