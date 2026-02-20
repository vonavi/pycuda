__global__ void add_kernel(const double *A, const double *B, double *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}
