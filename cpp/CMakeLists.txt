# NewToLLM v2 C++ side: one library mirroring py/newtollm, plus the CLI.
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_VENDOR_PATH})
find_package(Torch REQUIRED)

set(NEWTOLLM_SOURCES
    newtollm/report.cpp
    newtollm/config.cpp
    newtollm/tokenizer.cpp
    newtollm/tensors.cpp
    newtollm/autograd.cpp
    newtollm/modules.cpp
    newtollm/attention.cpp
    newtollm/gpt.cpp
    newtollm/data.cpp
    newtollm/train.cpp
)

add_library(newtollm_lib STATIC ${NEWTOLLM_SOURCES})
# Headers are included as "newtollm/<name>.h", same as the Python import path.
target_include_directories(newtollm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(newtollm_lib SYSTEM PRIVATE
    ${meta_pytorch_tokenizers_SOURCE_DIR}/third-party/json/single_include)
target_compile_definitions(newtollm_lib PUBLIC NEWTOLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# PUBLIC so executables inherit regex_lookahead's whole-archive link options: it registers
# the PCRE2 fallback (needed for GPT-2's `(?!\S)`) from a static initializer.
target_link_libraries(newtollm_lib PUBLIC ${TORCH_LIBRARIES} tokenizers::tokenizers)
target_compile_features(newtollm_lib PUBLIC cxx_std_20)
target_compile_options(newtollm_lib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(newtollm newtollm/main.cpp)
target_link_libraries(newtollm PRIVATE newtollm_lib)
set_target_properties(newtollm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(lesson tensors autograd modules attention gpt data train)
  add_test(NAME v2_${lesson} COMMAND newtollm ${lesson})
endforeach()
add_test(NAME v2_unknown_lesson COMMAND newtollm no_such_lesson)
set_tests_properties(v2_unknown_lesson PROPERTIES WILL_FAIL TRUE)

add_executable(parity tools/parity.cpp)
target_link_libraries(parity PRIVATE newtollm_lib)
set_target_properties(parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_test(NAME v2_parity_usage COMMAND parity)
set_tests_properties(v2_parity_usage PROPERTIES WILL_FAIL TRUE)
