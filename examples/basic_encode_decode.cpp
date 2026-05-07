#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_example_tokenizer() {
  const auto path =
      std::filesystem::temp_directory_path() / "tokenizers_cpp_example_basic.json";
  std::ofstream out(path);
  out << R"json({
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {"type": "WhitespaceSplit"},
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "tokenizers.cpp": 3
      }
    }
  })json";
  return path;
}

}  // namespace

int main() {
  const auto path = write_example_tokenizer();
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);

  const auto encoded = tokenizer.encode("hello tokenizers.cpp", false);
  assert((encoded.ids == std::vector<std::uint32_t>{1, 3}));
  assert((encoded.tokens == std::vector<std::string>{"hello", "tokenizers.cpp"}));
  assert((encoded.offsets ==
          std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 20}}));

  const auto decoded = tokenizer.decode(encoded.ids, true);
  assert(decoded == "hello tokenizers.cpp");

  std::cout << decoded << "\n";

  std::filesystem::remove(path);
  return 0;
}
