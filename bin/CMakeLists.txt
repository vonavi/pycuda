add_executable(launch launch.cu)
target_link_libraries(launch PRIVATE add_kernel)

install(TARGETS launch)
