cmake_minimum_required(VERSION 3.10)
project(LLM_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static deps (e.g. tokenizers) are linked into the shared my_torch library;
# required on Linux, harmless on macOS.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(cpp_impl)