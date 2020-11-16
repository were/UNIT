#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <cassert>
#include "cudnn.h"

#include "../util.h"

#define checkCudnnErr(x)                     \
  do {                                       \
    auto ret = x;                            \
    if (ret != CUDNN_STATUS_SUCCESS) {       \
      std::cerr << cudnnGetErrorString(ret); \
      assert(false);                         \
    }                                        \
  } while (false)

#define checkCudaErr(x)          \
  do {                           \
    auto ret = x;                \
    assert(ret == CUDA_SUCCESS); \
  } while (false)

int main() {
  cudnnHandle_t handle;

  checkCudnnErr(cudnnCreate(&handle));

  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;

  // Create your tensor descriptors:
  checkCudnnErr( cudnnCreateTensorDescriptor( &xDesc ));
  checkCudnnErr( cudnnCreateFilterDescriptor( &wDesc ));
  checkCudnnErr( cudnnCreateTensorDescriptor( &yDesc ));
  checkCudnnErr( cudnnCreateConvolutionDescriptor( &convDesc ));

  int dimA[4];
  int dimB[4];
  std::cin >> dimA[0] >> dimA[1] >> dimA[2] >> dimA[3];
  std::cin >> dimB[0] >> dimB[1] >> dimB[2] >> dimB[3];

  int pad[2] = {0, 0};
  int stride[2];
  int dilation[2] = {1, 1};

  std::cin >> stride[0] >> stride[1];

  int dimY[4] = {dimA[0], dimB[0], (dimA[2] - dimB[2]) / stride[0] + 1, (dimA[3] - dimB[3]) / stride[1] + 1};
  int dtype_x;
  std::cin >> dtype_x;
  cudnnDataType_t dtype = (cudnnDataType_t) dtype_x;

  checkCudnnErr( cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, dtype,
    dimA[0], dimA[1], dimA[2], dimA[3]) );
  assert(dimA[1] == dimB[1]);
  checkCudnnErr( cudnnSetFilter4dDescriptor(wDesc, dtype, CUDNN_TENSOR_NCHW,
    dimB[0], dimB[1], dimB[2], dimB[3]) );
  checkCudnnErr( cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, dtype,
    dimY[0], dimY[1], dimY[2], dimY[3]) );

  void *x = nullptr, *w = nullptr, *y = nullptr;
  // Allocate and initialize tensors (again, only the input tensor is shown):
  checkCudaErr(cudaMalloc(&x, dimA[0] * dimA[1] * dimA[2] * dimA[3] * sizeof(float)));
  checkCudaErr(cudaMalloc(&w, dimB[0] * dimB[1] * dimB[2] * dimB[3] * sizeof(float)));
  checkCudaErr(cudaMalloc(&y, dimY[0] * dimY[1] * dimY[2] * dimY[3] * sizeof(float)));


  // Set the compute data type (below as CUDNN_DATA_FLOAT):
  checkCudnnErr( cudnnSetConvolution2dDescriptor(convDesc,
                                                 pad[0], pad[1],
                                                 stride[0], stride[1],
                                                 dilation[0], dilation[1],
                                                 CUDNN_CROSS_CORRELATION, dtype));
  checkCudnnErr( cudnnSetConvolutionGroupCount(convDesc, 1) );

  int algo_x, tc;
  std::cin >> algo_x >> tc;

  if (tc) {
    // Set the math type to allow cuDNN to use Tensor Cores:
    checkCudnnErr( cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH) );
  }

  // Choose a supported algorithm:
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  algo = (cudnnConvolutionFwdAlgo_t) algo_x;

  size_t workSpaceSize;
  void *workSpace = nullptr;
  assert(handle);
  assert(xDesc);
  assert(wDesc);
  assert(yDesc);
  assert(convDesc);
  // Allocate your workspace:
  checkCudnnErr( cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, 
                                                         wDesc, convDesc,
                                                         yDesc, algo, &workSpaceSize) );

  if (workSpaceSize > 0) {
    cudaMalloc(&workSpace, workSpaceSize);
  }

  float alpha = 1.0;
  float beta = 0.0;

  assert(x);
  assert(w);
  assert(y);
  // Invoke the convolution:
  checkCudnnErr( cudnnConvolutionForward(handle, (void*)(&alpha),
                                         xDesc, x,
                                         wDesc, w, convDesc, algo,
                                         workSpace, workSpaceSize,
                                         (void*)(&beta),
                                         yDesc, y) );
  cudaDeviceSynchronize();
  begin_roi();
  for (int i = 0; i < 10; ++i) {
    checkCudnnErr( cudnnConvolutionForward(handle, (void*)(&alpha),
                                           xDesc, x,
                                           wDesc, w, convDesc, algo,
                                           workSpace, workSpaceSize,
                                           (void*)(&beta),
                                           yDesc, y) );
    checkCudaErr(cudaDeviceSynchronize());
  }
  double elps = end_roi();
  elps /= 10.;
  std::cout << "Exec: " << elps << "us" << std::endl;
  std::cout << ((double) dimY[0] * dimY[1] * dimY[2] * dimY[3] * dimB[1] * dimB[2] * dimB[3]) / elps / 1000.
            << " GFLOP/s" << std::endl;

  return 0;
}
