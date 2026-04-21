set(LIBTORCH_VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch)

include(FetchContent)
set(TOKENIZERS_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    meta_pytorch_tokenizers
    GIT_REPOSITORY https://github.com/meta-pytorch/tokenizers.git
    GIT_TAG v1.2.0
    GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(meta_pytorch_tokenizers)

add_subdirectory(my_torch)
add_subdirectory(sandbox)