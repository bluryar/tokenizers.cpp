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
      std::filesystem::temp_directory_path() / "tokenizers_cpp_example_stream.json";
  std::ofstream out(path);
  out << R"json({
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": {
      "type": "Sequence",
      "decoders": [
        {"type": "ByteFallback"},
        {"type": "Fuse"}
      ]
    },
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": {
        "[UNK]": 0,
        "<0xC3>": 1,
        "<0xA9>": 2,
        " token": 3
      }
    }
  })json";
  return path;
}

}  // namespace

int main() {
  const auto path = write_example_tokenizer();
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);

  assert(tokenizer.decode({1, 2, 3}, false) == "\xC3\xA9 token");

  auto stream = tokenizer.decode_stream(false);
  assert(!stream.has_pending());
  assert(!stream.step(1).has_value());
  assert(stream.has_pending());

  const auto first = stream.step(2);
  assert(first.has_value());
  assert(first.value() == "\xC3\xA9");
  assert(!stream.has_pending());

  const auto second = stream.step(3);
  assert(second.has_value());
  assert(second.value() == " token");
  assert(!stream.has_pending());

  std::cout << first.value() << second.value() << "\n";

  std::filesystem::remove(path);
  return 0;
}
