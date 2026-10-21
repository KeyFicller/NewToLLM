#pragma once
// Deterministic, language-neutral printing (twin of py/newtollm/report.py).
// Both sides print through `report` so their stdout can be diffed line by line.

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace newtollm::report {

inline constexpr int64_t kMaxValues = 8;

[[nodiscard]] std::string format_value(double value, c10::ScalarType dtype);
void section(std::string_view title);
void show(std::string_view label, const torch::Tensor& tensor);
void scalar(std::string_view label, double value);
void count(std::string_view label, int64_t value);
void text(std::string_view label, std::string_view value);

}  // namespace newtollm::report
