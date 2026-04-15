set(SOURCES
    my_torch.h
    my_torch_export.h
    verifier/torch_verifier.cpp
    brief/torch_breif.cpp
)

set(CMAKE_PREFIX_PATH ${LIBTORCH_VENDOR_PATH})
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS}/include)
include_directories(${TORCH_INCLUDE_DIRS}/include/torch/csrc/api/include)

add_library(my_torch SHARED ${SOURCES})
target_include_directories(my_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(my_torch PUBLIC ${TORCH_LIBRARIES})