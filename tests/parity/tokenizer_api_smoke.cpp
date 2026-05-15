#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

int main() {
  const auto path = std::filesystem::temp_directory_path() / "tokenizers_cpp_smoke_tokenizer.json";
  const std::string tokenizer_json = R"json({
    "version": "1.0",
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": {
        "[UNK]": 0,
        "[CLS]": 1,
        "hello": 2,
        "world": 3,
        "héllo": 4,
        "世界": 5
      }
    },
    "added_tokens": [
      {"id": 1, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    ]
  })json";
  {
    std::ofstream out(path);
    out << tokenizer_json;
  }

  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  const auto memory_tokenizer = tokenizers_cpp::Tokenizer::from_json(tokenizer_json);
  assert(memory_tokenizer.get_vocab_size() == tokenizer.get_vocab_size());
  assert((memory_tokenizer.encode("hello world").ids == tokenizer.encode("hello world").ids));
  assert(tokenizer.get_vocab_size() == 6);
  assert(tokenizer.token_to_id("hello").value() == 2);
  assert(tokenizer.id_to_token(3).value() == "world");

  const auto encoded = tokenizer.encode("hello world");
  assert((encoded.ids == std::vector<std::uint32_t>{2, 3}));
  assert((encoded.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((encoded.tokens == std::vector<std::string>{"hello", "world"}));
  assert((encoded.word_ids == std::vector<std::optional<std::uint32_t>>{0, 1}));
  assert((encoded.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((encoded.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(encoded.offsets.size() == 2);
  assert(encoded.offsets[0] == (tokenizers_cpp::Offset{0, 5}));
  assert(encoded.offsets[1] == (tokenizers_cpp::Offset{6, 11}));

  const auto unicode_byte = tokenizer.encode("héllo 世界", false);
  assert((unicode_byte.ids == std::vector<std::uint32_t>{4, 5}));
  assert((unicode_byte.offsets == std::vector<tokenizers_cpp::Offset>{{0, 6}, {7, 13}}));

  const auto unicode_char = tokenizer.encode_char_offsets("héllo 世界", false);
  assert((unicode_char.ids == std::vector<std::uint32_t>{4, 5}));
  assert((unicode_char.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 8}}));

  const auto pretokenized_char =
      tokenizer.encode_char_offsets(std::vector<std::string>{"héllo", "世界"}, false);
  assert((pretokenized_char.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));
  assert((pretokenized_char.word_ids == std::vector<std::optional<std::uint32_t>>{0, 1}));

  const auto pair_char = tokenizer.encode_pair_char_offsets("héllo", "世界", false);
  assert((pair_char.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));
  assert((pair_char.type_ids == std::vector<std::uint32_t>{0, 1}));

  const auto pretokenized_pair_char = tokenizer.encode_pair_char_offsets(
      std::vector<std::string>{"héllo"},
      std::vector<std::string>{"世界"},
      false);
  assert((pretokenized_pair_char.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));
  assert((pretokenized_pair_char.word_ids == std::vector<std::optional<std::uint32_t>>{0, 0}));

  const auto batch_char = tokenizer.encode_batch_char_offsets(
      std::vector<std::string>{"héllo", "héllo 世界"},
      false);
  assert(batch_char.size() == 2);
  assert((batch_char[0].offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}}));
  assert((batch_char[1].offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 8}}));

  const auto batch_pretokenized_char = tokenizer.encode_batch_char_offsets(
      std::vector<std::vector<std::string>>{{"héllo"}, {"héllo", "世界"}},
      false);
  assert(batch_pretokenized_char.size() == 2);
  assert((batch_pretokenized_char[0].offsets ==
          std::vector<tokenizers_cpp::Offset>{{0, 5}}));
  assert((batch_pretokenized_char[1].offsets ==
          std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));

  const auto batch_pair_char = tokenizer.encode_batch_pairs_char_offsets(
      std::vector<std::pair<std::string, std::string>>{{"héllo", "世界"}},
      false);
  assert(batch_pair_char.size() == 1);
  assert((batch_pair_char[0].offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));

  using PretokenizedPair =
      std::pair<std::vector<std::string>, std::vector<std::string>>;
  const auto batch_pretokenized_pair_char =
      tokenizer.encode_batch_pairs_char_offsets(
          std::vector<PretokenizedPair>{{{"héllo"}, {"世界"}}},
          false);
  assert(batch_pretokenized_pair_char.size() == 1);
  assert((batch_pretokenized_pair_char[0].offsets ==
          std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 2}}));

  const auto encoded_special = tokenizer.encode("[CLS] hello");
  assert((encoded_special.ids == std::vector<std::uint32_t>{1, 2}));
  assert((encoded_special.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((encoded_special.attention_mask == std::vector<std::uint32_t>{1, 1}));

  const auto pair = tokenizer.encode_pair("hello", "world");
  assert((pair.ids == std::vector<std::uint32_t>{2, 3}));
  assert((pair.type_ids == std::vector<std::uint32_t>{0, 1}));

  assert(tokenizer.decode({1, 2, 3}, true) == "hello world");
  assert(tokenizer.decode({1, 2, 3}, false) == "[CLS] hello world");
  assert(
      tokenizer.decode_batch({{1, 2, 3}, {2, 3}}, true) ==
      (std::vector<std::string>{"hello world", "hello world"}));
  assert(
      tokenizer.decode_batch({{1, 2, 3}, {2, 3}}, false) ==
      (std::vector<std::string>{"[CLS] hello world", "hello world"}));
  assert(tokenizer.decode_batch({}, true).empty());

  std::filesystem::remove(path);
  return 0;
}
