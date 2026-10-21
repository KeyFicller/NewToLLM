// CLI - mirrors py/newtollm/__main__.py.
//
//   newtollm <lesson> [--preset tiny|small|gpt2] [--seed N]
#include "newtollm/autograd.h"
#include "newtollm/config.h"
#include "newtollm/modules.h"
#include "newtollm/tensors.h"
#include "newtollm/tokenizer.h"

#include <array>
#include <exception>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct Lesson {
  std::string_view name;
  std::string_view description;
  std::function<void(const newtollm::Preset&)> run;
};

const std::array<Lesson, 3> kLessons = {{
    {.name = "tensors",
     .description = "01 creation, shapes, indexing, broadcasting",
     .run = newtollm::run_tensors},
    {.name = "autograd",
     .description = "02 gradients, no_grad, hand-written SGD",
     .run = newtollm::run_autograd},
    {.name = "modules",
     .description = "03 GELU, LayerNorm, FeedForward",
     .run = newtollm::run_modules},
}};

void tokenize_file(const std::string& path) {
  const std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const auto ids = newtollm::Tokenizer().encode(buffer.str());
  std::cout << ids.size() << '\n';
  for (size_t i = 0; i < ids.size(); ++i) {
    std::cout << (i == 0 ? "" : " ") << ids[i];
  }
  std::cout << '\n';
}

int usage(int code) {
  std::cerr << "usage: newtollm <lesson|list|tokenize FILE> [--preset NAME] [--seed N]\n";
  return code;
}

int run(int argc, char** argv) {
  std::optional<std::string> command;
  std::optional<std::string> path;
  std::string preset_name = "tiny";
  std::optional<uint64_t> seed;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--preset" && i + 1 < argc) {
      preset_name = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      seed = std::stoull(argv[++i]);
    } else if (arg.starts_with("-")) {
      return usage(2);
    } else if (!command) {
      command = std::string(arg);
    } else {
      path = std::string(arg);
    }
  }
  if (!command) {
    return usage(2);
  }
  if (*command == "list") {
    for (const auto& lesson : kLessons) {
      std::cout << std::format("{:<10} {}\n", lesson.name, lesson.description);
    }
    return 0;
  }
  if (*command == "tokenize") {
    if (!path) {
      return usage(2);
    }
    tokenize_file(*path);
    return 0;
  }
  auto preset = newtollm::load_preset(preset_name);
  if (seed) {
    preset = preset.with_seed(*seed);
  }
  for (const auto& lesson : kLessons) {
    if (lesson.name == *command) {
      lesson.run(preset);
      return 0;
    }
  }
  return usage(2);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
