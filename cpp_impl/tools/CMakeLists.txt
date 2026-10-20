add_executable(parity_check parity_check.cpp)
target_link_libraries(parity_check my_torch)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark my_torch)
