add_executable(launch launch.cpp)
target_link_libraries(launch PRIVATE add_kernel)

install(TARGETS launch)
