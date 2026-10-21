#pragma once
// Lesson 02 - autograd (twin of py/newtollm/autograd.py).

#include "newtollm/config.h"

#include <cstdint>

namespace newtollm {

void gradients();
void no_grad();
void fit_line(uint64_t seed, int64_t steps = 50, double lr = 0.1);
void run_autograd(const Preset& preset);

}  // namespace newtollm
