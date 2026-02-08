add_executable(add_kernel add_kernel.cu)
set_target_properties(add_kernel PROPERTIES
  CUDA_ARCHITECTURES 61
)

install(TARGETS add_kernel)
