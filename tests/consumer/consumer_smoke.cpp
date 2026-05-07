#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#ifndef TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON
#error "TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON must be defined"
#endif

int main() {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(TOKENIZERS_CPP_CONSUMER_TOKENIZER_JSON);

  assert(tokenizer.get_vocab_size() == 5);
  assert(tokenizer.token_to_id("hello").value() == 2);
  assert(tokenizer.id_to_token(3).value() == "world");

  const auto encoded = tokenizer.encode("hello world", false);
  assert((encoded.ids == std::vector<std::uint32_t>{2, 3}));
  assert((encoded.tokens == std::vector<std::string>{"hello", "world"}));
  assert((encoded.offsets ==
          std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 11}}));
  assert((encoded.word_ids ==
          std::vector<std::optional<std::uint32_t>>{0, 1}));
  assert((encoded.attention_mask == std::vector<std::uint32_t>{1, 1}));

  const auto pair = tokenizer.encode_pair("hello", "goodbye", false);
  assert((pair.ids == std::vector<std::uint32_t>{2, 4}));
  assert((pair.type_ids == std::vector<std::uint32_t>{0, 1}));

  const auto batch = tokenizer.encode_batch(
      std::vector<std::string>{"hello world", "goodbye"},
      false);
  assert(batch.size() == 2);
  assert((batch[0].ids == std::vector<std::uint32_t>{2, 3}));
  assert((batch[1].ids == std::vector<std::uint32_t>{4}));

  assert(tokenizer.decode({2, 3}, true) == "hello world");
  assert(
      tokenizer.decode_batch({{2, 3}, {4}}, true) ==
      (std::vector<std::string>{"hello world", "goodbye"}));

  return 0;
}
