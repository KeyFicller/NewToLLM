# LibTorch location: defaults to cpp_impl/vendor/libtorch, override with
#   cmake -DLIBTORCH_VENDOR_PATH=/path/to/libtorch ...
set(LIBTORCH_VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch CACHE PATH "Path to extracted LibTorch")

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

add_subdirectory(my_torch)
add_subdirectory(sandbox)
add_subdirectory(tools)

# CTest: smoke tests for the C++ side (run from repo root so assets/ resolve).
enable_testing()
set(NEWTOLLM_ROOT ${PROJECT_SOURCE_DIR})
add_test(NAME sandbox_list COMMAND sandbox list WORKING_DIRECTORY ${NEWTOLLM_ROOT})
add_test(NAME sandbox_verify COMMAND sandbox verify --device cpu WORKING_DIRECTORY ${NEWTOLLM_ROOT})
add_test(NAME sandbox_brief COMMAND sandbox brief WORKING_DIRECTORY ${NEWTOLLM_ROOT})
add_test(NAME sandbox_toy_model_small COMMAND sandbox toy_model --small WORKING_DIRECTORY ${NEWTOLLM_ROOT})
add_test(NAME sandbox_unknown_task COMMAND sandbox no_such_task WORKING_DIRECTORY ${NEWTOLLM_ROOT})
set_tests_properties(sandbox_unknown_task PROPERTIES WILL_FAIL TRUE)
set_tests_properties(sandbox_verify PROPERTIES PASS_REGULAR_EXPRESSION "device should be: cpu")
