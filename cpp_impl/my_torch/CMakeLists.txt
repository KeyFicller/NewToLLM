set(SOURCES
    my_torch.h
    my_torch_export.h
    verifier/torch_verifier.cpp
    basic/torch_brief.cpp
    toy_model/model.cpp
    toy_model/transformer.cpp
    toy_model/torch_toy_model.cpp
    load_model/torch_load_model.cpp
    train/data_utils.cpp
    train/torch_train.cpp
    fine_tuning/fine_tuning_classify.cpp
    fine_tuning/torch_fine_tuning.cpp
    utils/torch_utils.cpp
    utils/runtime.cpp
)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_VENDOR_PATH})
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS}/include)
include_directories(${TORCH_INCLUDE_DIRS}/include/torch/csrc/api/include)

add_library(my_torch SHARED ${SOURCES})
target_include_directories(my_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(my_torch PUBLIC ${TORCH_LIBRARIES} tokenizers::tokenizers)
# Shared config parsing (configs/toy_gpt2.json) uses nlohmann/json vendored by tokenizers.
target_include_directories(my_torch PRIVATE ${meta_pytorch_tokenizers_SOURCE_DIR}/third-party/json/single_include)
target_compile_definitions(my_torch PRIVATE NEWTOLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
