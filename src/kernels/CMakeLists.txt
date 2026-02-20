add_library(add_kernel_ptx OBJECT add_kernel.cu)
set_target_properties(add_kernel_ptx PROPERTIES
  CUDA_ARCHITECTURES 61
  CUDA_PTX_COMPILATION ON
)

install(
  FILES "$<TARGET_OBJECTS:add_kernel_ptx>"
  DESTINATION kernels
)
