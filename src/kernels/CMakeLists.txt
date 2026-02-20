add_library(add_kernel STATIC add_kernel.cu)
set_target_properties(add_kernel PROPERTIES
  CUDA_ARCHITECTURES 61
)
