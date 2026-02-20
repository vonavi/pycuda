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

  // Load a kernel
  CUmodule module;
  CUDA_CHECK(cuModuleLoad(&module, ptxFile.c_str()));
  CUfunction function;
  CUDA_CHECK(cuModuleGetFunction(&function, module, funcName.c_str()));

  // Allocate device memory
  CUdeviceptr dev_A, dev_B, dev_C;
  CUDA_CHECK(cuMemAlloc(&dev_A, bytes));
  CUDA_CHECK(cuMemAlloc(&dev_B, bytes));
  CUDA_CHECK(cuMemAlloc(&dev_C, bytes));

  // Initialize a stream to execute the kernel
  CUstream stream;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // Fill arrays on the host
  std::vector<double> host_A(N), host_B(N);
  for (int i = 0; i < N; ++i) {
    host_A[i] = i;
    host_B[i] = 2 * i;
  }

  // Copy data to the device
  CUDA_CHECK(cuMemcpyHtoDAsync(dev_A, host_A.data(), bytes, stream));
  CUDA_CHECK(cuMemcpyHtoDAsync(dev_B, host_B.data(), bytes, stream));

  // Launch the kernel
  void *args[] = {&dev_A, &dev_B, &dev_C, (void *)&N};
  CUDA_CHECK(cuLaunchKernel(function,  //
                            1, 1, 1,   // gridDim
                            256, 1, 1, // blockDim
                            0, stream, args, nullptr));

  // Copy the return data back
  std::vector<double> host_C(N);
  CUDA_CHECK(cuMemcpyDtoHAsync(host_C.data(), dev_C, bytes, stream));

  CUDA_CHECK(cuStreamSynchronize(stream));
  CUDA_CHECK(cuStreamDestroy(stream));

  // Cleanup
  CUDA_CHECK(cuMemFree(dev_A));
  CUDA_CHECK(cuMemFree(dev_B));
  CUDA_CHECK(cuMemFree(dev_C));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));

  for (int i = 0; i < N; ++i)
    std::cout << host_A[i] << " + " << host_B[i] << " = " << host_C[i]
              << std::endl;

  return EXIT_SUCCESS;
}
