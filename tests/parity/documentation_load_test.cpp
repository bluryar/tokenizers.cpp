#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef TOKENIZERS_CPP_HF_TEST_DATA_DIR
#error "TOKENIZERS_CPP_HF_TEST_DATA_DIR must be defined"
#endif

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");

  const std::string example = "This is an example";
  const std::vector<std::uint32_t> ids = {713, 16, 41, 1246};
  const std::vector<std::string> tokens = {
      "This",
      "\xC4\xA0"
      "is",
      "\xC4\xA0"
      "an",
      "\xC4\xA0"
      "example",
  };

  const auto encoding = tokenizer.encode(example, false);
  assert(encoding.ids == ids);
  assert(encoding.tokens == tokens);
  assert(tokenizer.decode(ids, false) == example);

  return 0;
}
