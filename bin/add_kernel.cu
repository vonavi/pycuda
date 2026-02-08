#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

// Error handling macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " ("     \
                << #call << "): " << cudaGetErrorString(err) << std::endl;     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Kernel
__global__ void add_kernel(const double *A, const double *B, double *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

int main() {
  const int N = 16;
  const size_t bytes = N * sizeof(double);

  int devCount;
  CUDA_CHECK(cudaGetDeviceCount(&devCount));
  if (0 == devCount) {
    std::cerr << "There is no CUDA capable device!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cudaSetDevice(0));

  // Allocate memory for arrays A, B, and C on host
  std::vector<double> h_A(N), h_B(N), h_C(N);
  // Fill host arrays A and B
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = 2 * i;
  }

  // Allocate memory for arrays d_A, d_B, and d_C on device
  double *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  // Copy data from host arrays A and B to device arrays d_A and d_B
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  // Launch kernel
  add_kernel<<<1, 256>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy data from device array d_C to host array C
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i)
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;

  // Free GPU memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return EXIT_SUCCESS;
}
