#include <cuda.h>
#include <iostream>
#include <cassert>
#include "cudnn.h"

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
  int strideA[4] = {0, 0, 0, 1};
  for (int i = 2; i >= 0; --i) {
    strideA[i] = strideA[i + 1] * dimA[i];
  }

  int pad[2] = {0};
  int stride[2];
  int dilation[2] = {1, 1};

  std::cin >> stride[0] >> stride[1];

  int dimY[4] = {dimA[0], dimB[0], (dimA[2] - dimB[2]) / stride[0] + 1, (dimA[3] - dimB[3]) / stride[1] + 1};
  int strideY[4] = {0, 0, 0, 1};
  for (int i = 2; i >= 0; --i) {
    strideY[i] = strideY[i + 1] * dimY[i];
  }

  checkCudnnErr( cudnnSetConvolution2dDescriptor(convDesc, pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) );

  checkCudnnErr( cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dimA[0], dimA[1], dimA[2], dimA[3]) );
  checkCudnnErr( cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dimB[0], dimB[1], dimB[2], dimB[3]) );
  checkCudnnErr( cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dimY[0], dimY[1], dimY[2], dimY[3]) );

  void *x, *w, *y;
  // Allocate and initialize tensors (again, only the input tensor is shown):
  checkCudaErr(cudaMalloc(&x, strideA[0] * sizeof(float)));
  checkCudaErr(cudaMalloc(&w, dimB[0] * dimB[1] * dimB[2] * dimB[3] * sizeof(float)));
  checkCudaErr(cudaMalloc(&w, strideY[0] * sizeof(float)));


  // Set the compute data type (below as CUDNN_DATA_FLOAT):
  checkCudnnErr( cudnnSetConvolution2dDescriptor(convDesc,
                                                 pad[0], pad[1],
                                                 stride[0], stride[1],
                                                 dilation[0], dilation[1],
                                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Set the math type to allow cuDNN to use Tensor Cores:
  checkCudnnErr( cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH) );

  // Choose a supported algorithm:
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;

  size_t workSpaceSize;
  void *workSpace;
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

  // Invoke the convolution:
  checkCudnnErr( cudnnConvolutionForward(handle, (void*)(&alpha), xDesc, x,
                                         wDesc, w, convDesc, algo,
                                         workSpace, workSpaceSize, (void*)(&beta),
                                         yDesc, y) );

  return 0;
}