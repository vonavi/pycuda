add_executable(launch launch.cpp)
target_link_libraries(launch PRIVATE CUDA::cuda_driver)
add_dependencies(launch add_kernel_ptx)

install(TARGETS launch)
