#pragma once

#ifdef _MSC_VER
#define MY_TORCH_EXPORT __declspec(dllexport)
#define MY_TORCH_IMPORT __declspec(dllimport)
#else
#define MY_TORCH_EXPORT __attribute__((visibility("default")))
#define MY_TORCH_IMPORT __attribute__((visibility("default")))
#endif

#ifdef MY_TORCH_BUILD
#define MY_TORCH_API MY_TORCH_EXPORT
#else
#define MY_TORCH_API MY_TORCH_IMPORT
#endif
