#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

std::filesystem::path write_temp_tokenizer_json(const std::string & name, const json & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

json wordlevel_truncation_tokenizer_json(
    json truncation,
    json post_processor = nullptr) {
  return {
      {"version", "1.0"},
      {"truncation", std::move(truncation)},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", {{"type", "WhitespaceSplit"}}},
      {"post_processor", std::move(post_processor)},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "WordLevel"},
           {"unk_token", "[UNK]"},
           {"vocab",
            {
                {"[UNK]", 0},
                {"one", 1},
                {"two", 2},
                {"three", 3},
                {"four", 4},
                {"five", 5},
                {"[CLS]", 101},
                {"[SEP]", 102},
            }},
       }},
  };
}

tokenizers_cpp::Tokenizer load_tokenizer(const std::string & name, const json & value) {
  const auto path = write_temp_tokenizer_json(name, value);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

std::vector<tokenizers_cpp::Offset> offsets(
    std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

json truncation(
    std::uint32_t max_length,
    std::uint32_t stride,
    const std::string & direction = "Right",
    const std::string & strategy = "LongestFirst") {
  return {
      {"direction", direction},
      {"max_length", max_length},
      {"strategy", strategy},
      {"stride", stride},
  };
}

void test_right_truncation_with_stride_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_right_stride.json",
      wordlevel_truncation_tokenizer_json(truncation(4, 2)));

  const auto output = tokenizer.encode("one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 4}));
  assert((output.tokens == std::vector<std::string>{"one", "two", "three", "four"}));
  assert((output.offsets == offsets({{0, 3}, {4, 7}, {8, 13}, {14, 18}})));
  assert((output.word_ids == word_ids({0U, 1U, 2U, 3U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing[0];
  assert((overflow.ids == std::vector<std::uint32_t>{3, 4, 5}));
  assert((overflow.tokens == std::vector<std::string>{"three", "four", "five"}));
  assert((overflow.offsets == offsets({{8, 13}, {14, 18}, {19, 23}})));
  assert((overflow.word_ids == word_ids({2U, 3U, 4U})));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_left_truncation_keeps_tail_tokens() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_left.json",
      wordlevel_truncation_tokenizer_json(truncation(3, 0, "Left")));

  const auto output = tokenizer.encode("one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{3, 4, 5}));
  assert((output.tokens == std::vector<std::string>{"three", "four", "five"}));
  assert((output.offsets == offsets({{8, 13}, {14, 18}, {19, 23}})));
  assert((output.word_ids == word_ids({2U, 3U, 4U})));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing[0];
  assert((overflow.ids == std::vector<std::uint32_t>{1, 2}));
  assert((overflow.tokens == std::vector<std::string>{"one", "two"}));
  assert((overflow.offsets == offsets({{0, 3}, {4, 7}})));
  assert((overflow.word_ids == word_ids({0U, 1U})));
}

void test_deserialize_old_truncation_defaults_to_right() {
  auto value = wordlevel_truncation_tokenizer_json({
      {"max_length", 2},
      {"strategy", "LongestFirst"},
      {"stride", 0},
  });
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_default_direction.json",
      value);

  const auto output = tokenizer.encode("one two three", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert(output.overflowing.size() == 1);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{3}));
}

void test_truncation_accounts_for_single_special_tokens() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_bert_specials.json",
      wordlevel_truncation_tokenizer_json(
          truncation(5, 0),
          {{"type", "BertProcessing"},
           {"sep", {"[SEP]", 102}},
           {"cls", {"[CLS]", 101}}}));

  const auto output = tokenizer.encode("one two three four five", true);
  assert((output.ids == std::vector<std::uint32_t>{101, 1, 2, 3, 102}));
  assert((output.tokens == std::vector<std::string>{"[CLS]", "one", "two", "three", "[SEP]"}));
  assert((output.offsets == offsets({{0, 0}, {0, 3}, {4, 7}, {8, 13}, {0, 0}})));
  assert((output.word_ids == word_ids({std::nullopt, 0U, 1U, 2U, std::nullopt})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing[0];
  assert((overflow.ids == std::vector<std::uint32_t>{101, 4, 5, 102}));
  assert((overflow.tokens == std::vector<std::string>{"[CLS]", "four", "five", "[SEP]"}));
  assert((overflow.offsets == offsets({{0, 0}, {14, 18}, {19, 23}, {0, 0}})));
  assert((overflow.word_ids == word_ids({std::nullopt, 3U, 4U, std::nullopt})));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_invalid_stride_rejected_on_load() {
  try {
    (void)load_tokenizer(
        "tokenizers_cpp_truncation_invalid_stride.json",
        wordlevel_truncation_tokenizer_json(truncation(2, 2)));
    assert(false && "invalid truncation stride should fail");
  } catch (const std::runtime_error & error) {
    assert(std::string(error.what()).find("stride") != std::string::npos);
  }
}

void test_only_second_single_sequence_rejected_when_truncated() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_only_second_single.json",
      wordlevel_truncation_tokenizer_json({
          {"direction", "Right"},
          {"max_length", 2},
          {"strategy", "OnlySecond"},
          {"stride", 0},
      }));

  try {
    (void)tokenizer.encode("one two three", false);
    assert(false && "OnlySecond single-sequence truncation should fail");
  } catch (const std::runtime_error & error) {
    assert(std::string(error.what()).find("OnlySecond") != std::string::npos);
  }
}

void test_pair_longest_first_overflow_merge_order() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_pair_longest_first.json",
      wordlevel_truncation_tokenizer_json(truncation(7, 0)));

  const auto output =
      tokenizer.encode_pair("one two three four five", "one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 1, 2, 3, 4}));
  assert((output.tokens == std::vector<std::string>{
                               "one",
                               "two",
                               "three",
                               "one",
                               "two",
                               "three",
                               "four",
                           }));
  assert((output.offsets == offsets(
              {{0, 3}, {4, 7}, {8, 13}, {0, 3}, {4, 7}, {8, 13}, {14, 18}})));
  assert((output.word_ids == word_ids({0U, 1U, 2U, 0U, 1U, 2U, 3U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 1, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1}));

  assert(output.overflowing.size() == 3);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{4, 5, 1, 2, 3, 4}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 1, 1}));
  assert((output.overflowing[1].ids == std::vector<std::uint32_t>{4, 5, 5}));
  assert((output.overflowing[1].type_ids == std::vector<std::uint32_t>{0, 0, 1}));
  assert((output.overflowing[2].ids == std::vector<std::uint32_t>{1, 2, 3, 5}));
  assert((output.overflowing[2].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1}));
}

void test_pair_only_second_truncates_second_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_pair_only_second.json",
      wordlevel_truncation_tokenizer_json(
          truncation(5, 0, "Right", "OnlySecond")));

  const auto output =
      tokenizer.encode_pair("one two three", "one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 1, 2}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1}));
  assert((output.word_ids == word_ids({0U, 1U, 2U, 0U, 1U})));
  assert(output.overflowing.size() == 2);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{1, 2, 3, 3, 4}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1}));
  assert((output.overflowing[1].ids == std::vector<std::uint32_t>{1, 2, 3, 5}));
  assert((output.overflowing[1].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1}));
}

void test_pair_only_first_truncates_first_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_pair_only_first.json",
      wordlevel_truncation_tokenizer_json(
          truncation(5, 0, "Right", "OnlyFirst")));

  const auto output =
      tokenizer.encode_pair("one two three four five", "one two three", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 1, 2, 3}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 1}));
  assert((output.word_ids == word_ids({0U, 1U, 0U, 1U, 2U})));
  assert(output.overflowing.size() == 2);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{3, 4, 1, 2, 3}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 1}));
  assert((output.overflowing[1].ids == std::vector<std::uint32_t>{5, 1, 2, 3}));
  assert((output.overflowing[1].type_ids == std::vector<std::uint32_t>{0, 1, 1, 1}));
}

void test_pair_truncation_accounts_for_special_tokens() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_pair_bert_specials.json",
      wordlevel_truncation_tokenizer_json(
          truncation(7, 0),
          {{"type", "BertProcessing"},
           {"sep", {"[SEP]", 102}},
           {"cls", {"[CLS]", 101}}}));

  const auto output = tokenizer.encode_pair("one two three", "one two three", true);
  assert((output.ids == std::vector<std::uint32_t>{101, 1, 2, 102, 1, 2, 102}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "one",
                               "two",
                               "[SEP]",
                               "one",
                               "two",
                               "[SEP]",
                           }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1, 0, 0, 1}));
  assert((output.word_ids == word_ids({std::nullopt, 0U, 1U, std::nullopt, 0U, 1U, std::nullopt})));

  assert(output.overflowing.size() == 3);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{101, 3, 102, 1, 2, 102}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 1}));
  assert((output.overflowing[0].special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 0, 1}));
  assert((output.overflowing[1].ids == std::vector<std::uint32_t>{101, 3, 102, 3, 102}));
  assert((output.overflowing[1].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1}));
  assert((output.overflowing[1].special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((output.overflowing[2].ids == std::vector<std::uint32_t>{101, 1, 2, 102, 3, 102}));
  assert((output.overflowing[2].type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1}));
  assert((output.overflowing[2].special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1, 0, 1}));
}

void test_pair_only_first_too_short_rejected() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_truncation_pair_only_first_too_short.json",
      wordlevel_truncation_tokenizer_json(
          truncation(2, 0, "Right", "OnlyFirst")));

  try {
    (void)tokenizer.encode_pair("one two", "one two three four five", false);
    assert(false && "OnlyFirst should fail when the target cannot shrink enough");
  } catch (const std::runtime_error & error) {
    assert(std::string(error.what()).find("too short") != std::string::npos);
  }
}

}  // namespace

int main() {
  test_right_truncation_with_stride_overflowing();
  test_left_truncation_keeps_tail_tokens();
  test_deserialize_old_truncation_defaults_to_right();
  test_truncation_accounts_for_single_special_tokens();
  test_invalid_stride_rejected_on_load();
  test_only_second_single_sequence_rejected_when_truncated();
  test_pair_longest_first_overflow_merge_order();
  test_pair_only_second_truncates_second_sequence();
  test_pair_only_first_truncates_first_sequence();
  test_pair_truncation_accounts_for_special_tokens();
  test_pair_only_first_too_short_rejected();
  return 0;
}
