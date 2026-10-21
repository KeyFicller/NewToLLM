cmake_minimum_required(VERSION 3.24)
project(TorchTwins LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# LibTorch location: defaults to vendor/libtorch, override with
#   cmake -DLIBTORCH_VENDOR_PATH=/path/to/libtorch ...
set(LIBTORCH_VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch CACHE PATH "Path to extracted LibTorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_VENDOR_PATH})

# GPT-2 BPE tokenizer (tiktoken format) for the C++ side.
include(FetchContent)
set(TOKENIZERS_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
# GPT-2's pre-tokenizer regex uses a lookahead `\s+(?!\S)`; RE2 cannot do that.
set(SUPPORT_REGEX_LOOKAHEAD ON CACHE BOOL "" FORCE)
set(TOKENIZERS_ENABLE_LOGGING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    meta_pytorch_tokenizers
    GIT_REPOSITORY https://github.com/meta-pytorch/tokenizers.git
    GIT_TAG v1.2.0
    GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(meta_pytorch_tokenizers)

enable_testing()
add_subdirectory(cpp)
