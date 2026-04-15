add_executable(sandbox main.cpp)

target_link_libraries(sandbox my_torch)
target_include_directories(sandbox PUBLIC ${MY_TORCH_INCLUDE_DIR})