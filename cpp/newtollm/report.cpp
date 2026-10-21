#include "newtollm/report.h"

#include <format>
#include <iostream>
#include <vector>

namespace newtollm::report {

// @twin format_value
std::string format_value(double value, c10::ScalarType dtype) {
  if (dtype == torch::kBool) {
    return value != 0.0 ? "true" : "false";
  }
  if (c10::isFloatingType(dtype)) {
    return std::format("{:.4f}", value);
  }
  return std::format("{}", static_cast<int64_t>(value));
}

// @twin section
void section(std::string_view title) { std::cout << std::format("== {} ==\n", title); }

// @twin show
void show(std::string_view label, const torch::Tensor& tensor) {
  const auto flat = tensor.detach().flatten().cpu();
  const auto as_double = flat.to(torch::kDouble);
  std::string head;
  const int64_t shown = std::min(flat.numel(), kMaxValues);
  for (int64_t i = 0; i < shown; ++i) {
    head += (i == 0 ? "" : ", ") + format_value(as_double[i].item<double>(), flat.scalar_type());
  }
  if (flat.numel() > kMaxValues) {
    head += ", ...";
  }
  const bool is_bool = flat.scalar_type() == torch::kBool;
  const auto total = is_bool ? flat.sum(torch::kLong) : flat.sum();
  const auto sum_dtype = is_bool ? torch::kLong : flat.scalar_type();
  std::string shape;
  for (const auto dim : tensor.sizes()) {
    shape += (shape.empty() ? "" : ", ") + std::to_string(dim);
  }
  std::cout << std::format("{}: shape=[{}] values=[{}] sum={}\n", label, shape, head,
                           format_value(total.item<double>(), sum_dtype));
}

// @twin scalar
void scalar(std::string_view label, double value) {
  std::cout << std::format("{}: {:.4f}\n", label, value);
}

// @twin count
void count(std::string_view label, int64_t value) {
  std::cout << std::format("{}: {}\n", label, value);
}

// @twin text
void text(std::string_view label, std::string_view value) {
  std::cout << std::format("{}: {}\n", label, value);
}

}  // namespace newtollm::report
