#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_example_tokenizer() {
  const auto path =
      std::filesystem::temp_directory_path() / "tokenizers_cpp_example_batch.json";
  std::ofstream out(path);
  out << R"json({
    "version": "1.0",
    "truncation": null,
    "padding": {
      "strategy": "BatchLongest",
      "direction": "Right",
      "pad_to_multiple_of": null,
      "pad_id": 0,
      "pad_type_id": 0,
      "pad_token": "[PAD]"
    },
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {"type": "WhitespaceSplit"},
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": {
        "[PAD]": 0,
        "[UNK]": 99,
        "one": 1,
        "two": 2,
        "three": 3
      }
    }
  })json";
  return path;
}

}  // namespace

int main() {
  const auto path = write_example_tokenizer();
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);

  const auto batch = tokenizer.encode_batch(
      std::vector<std::string>{"one two three", "one"},
      false);
  assert(batch.size() == 2);
  assert((batch[0].ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert((batch[0].attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
  assert((batch[1].ids == std::vector<std::uint32_t>{1, 0, 0}));
  assert((batch[1].tokens == std::vector<std::string>{"one", "[PAD]", "[PAD]"}));
  assert((batch[1].word_ids ==
          std::vector<std::optional<std::uint32_t>>{
              std::optional<std::uint32_t>{0},
              std::nullopt,
              std::nullopt,
          }));
  assert((batch[1].attention_mask == std::vector<std::uint32_t>{1, 0, 0}));

  std::cout << "batch size: " << batch.size()
            << ", padded length: " << batch[1].ids.size() << "\n";

  std::filesystem::remove(path);
  return 0;
}
