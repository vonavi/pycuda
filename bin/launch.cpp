#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

#include <cuda.h>

// Error handling macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult result = call;                                                    \
    if (CUDA_SUCCESS != result) {                                              \
      const char *msg;                                                         \
      cuGetErrorString(result, &msg);                                          \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " ("     \
                << #call << "): " << msg << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <ptx_file> <func_name>" << std::endl;
    return EXIT_FAILURE;
  }

  std::filesystem::path ptxFile(argv[1]);
  std::string funcName(argv[2]);
  if (!std::filesystem::exists(ptxFile)) {
    std::cerr << "Failed to open file: " << ptxFile << std::endl;
    return EXIT_FAILURE;
  }

  const int N = 16;
  const size_t bytes = N * sizeof(double);

  CUDA_CHECK(cuInit(0));

  int devCount;
  CUDA_CHECK(cuDeviceGetCount(&devCount));
  if (0 == devCount) {
    std::cerr << "There is no CUDA capable device!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, 0, device));

  // Load kernel
  CUmodule module;
  CUDA_CHECK(cuModuleLoad(&module, ptxFile.c_str()));
  CUfunction function;
  CUDA_CHECK(cuModuleGetFunction(&function, module, funcName.c_str()));

  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // Allocate memory for arrays A, B, and C on host
  std::vector<double> h_A(N), h_B(N), h_C(N);
  // Fill host arrays A and B
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = 2 * i;
  }

  // Allocate memory for arrays d_A, d_B, and d_C on device
  CUdeviceptr d_A, d_B, d_C;
  CUDA_CHECK(cuMemAlloc(&d_A, bytes));
  CUDA_CHECK(cuMemAlloc(&d_B, bytes));
  CUDA_CHECK(cuMemAlloc(&d_C, bytes));

  // Copy data from host arrays A and B to device arrays d_A and d_B
  CUDA_CHECK(cuMemcpyHtoDAsync(d_A, h_A.data(), bytes, stream));
  CUDA_CHECK(cuMemcpyHtoDAsync(d_B, h_B.data(), bytes, stream));

  // Launch kernel
  void *args[] = {&d_A, &d_B, &d_C, (void *)&N};
  CUDA_CHECK(cuLaunchKernel(function,  //
                            1, 1, 1,   // gridDim
                            256, 1, 1, // blockDim
                            0, stream, args, nullptr));

  // Copy data from device array d_C to host array C
  CUDA_CHECK(cuMemcpyDtoHAsync(h_C.data(), d_C, bytes, stream));

  CUDA_CHECK(cuStreamSynchronize(stream));
  CUDA_CHECK(cuStreamDestroy(stream));

  // Cleanup
  CUDA_CHECK(cuMemFree(d_A));
  CUDA_CHECK(cuMemFree(d_B));
  CUDA_CHECK(cuMemFree(d_C));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));

  for (int i = 0; i < N; ++i)
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;

  return EXIT_SUCCESS;
}
