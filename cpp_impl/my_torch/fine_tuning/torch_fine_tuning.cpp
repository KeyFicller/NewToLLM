#include "fine_tuning/fine_tuning_classify.h"
#include "my_torch.h"

#include <iostream>

namespace MyTorch {

void fine_tuning_classify_torch() { fine_tuning_classify(); }

void fine_tuning_instruction_torch() {
  // Python: python_impl/fine_tuning/fine_tuning_instruction.py
  std::cout << "[fine_tuning_instruction] not implemented on the C++ side yet "
               "(Python only, see docs/REFACTOR_PLAN.md)\n";
}

} // namespace MyTorch
