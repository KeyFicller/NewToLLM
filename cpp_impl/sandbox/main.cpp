// C++ (LibTorch) entrypoint.
//
// Task names are kept identical to the Python (PyTorch) entrypoint `main.py`,
// so the same experiment can be run on both sides:
//
//   python main.py <task>
//   ./build/cpp_impl/sandbox/sandbox <task>
#include "my_torch.h"

#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Task {
  std::string name;
  std::string description;
  std::function<void()> run;
};

const std::vector<Task> &tasks() {
  static const std::vector<Task> kTasks = {
      {"verify", "torch version / device check", MyTorch::verify_torch},
      {"brief", "torch basic examples", MyTorch::brief_torch},
      {"toy_model", "toy model generation example", MyTorch::toy_model_torch},
      {"train", "toy model pre-training example", MyTorch::train_torch},
      {"load_model", "load public GPT-2 weights into toy model",
       MyTorch::load_model_torch},
      {"fine_tuning", "fine-tuning example", MyTorch::fine_tuning_torch},
  };
  return kTasks;
}

void run_task(const Task &task) {
  std::cout << "--- [" << task.name << "] " << task.description << " ---\n";
  task.run();
}

void print_usage(const char *prog) {
  std::cout << "usage: " << prog << " [task]\n\n"
            << "task (default: verify):\n";
  for (const auto &task : tasks()) {
    std::cout << "  " << task.name << std::string(13 - task.name.size(), ' ')
              << task.description << '\n';
  }
  std::cout << "  all          run every task in order\n"
            << "  list         print tasks\n";
}

} // namespace

int main(int argc, char **argv) {
  const std::string name = argc > 1 ? argv[1] : "verify";

  if (name == "-h" || name == "--help" || name == "list") {
    print_usage(argv[0]);
    return 0;
  }
  if (name == "all") {
    for (const auto &task : tasks()) {
      run_task(task);
    }
    return 0;
  }
  for (const auto &task : tasks()) {
    if (task.name == name) {
      run_task(task);
      return 0;
    }
  }
  std::cerr << "unknown task: " << name << "\n\n";
  print_usage(argv[0]);
  return 2;
}
