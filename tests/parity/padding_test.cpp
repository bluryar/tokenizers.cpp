#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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

json wordlevel_tokenizer_json(
    json padding,
    json truncation = nullptr,
    json post_processor = nullptr) {
  return {
      {"version", "1.0"},
      {"truncation", std::move(truncation)},
      {"padding", std::move(padding)},
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
                {"[PAD]", 0},
                {"one", 1},
                {"two", 2},
                {"three", 3},
                {"four", 4},
                {"five", 5},
                {"[UNK]", 99},
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

json fixed_padding(
    std::uint32_t size,
    const std::string & direction = "Right",
    json pad_to_multiple_of = nullptr,
    std::uint32_t pad_type_id = 0) {
  return {
      {"strategy", {{"Fixed", size}}},
      {"direction", direction},
      {"pad_to_multiple_of", std::move(pad_to_multiple_of)},
      {"pad_id", 0},
      {"pad_type_id", pad_type_id},
      {"pad_token", "[PAD]"},
  };
}

json batch_longest_padding(json pad_to_multiple_of = nullptr) {
  return {
      {"strategy", "BatchLongest"},
      {"direction", "Right"},
      {"pad_to_multiple_of", std::move(pad_to_multiple_of)},
      {"pad_id", 0},
      {"pad_type_id", 0},
      {"pad_token", "[PAD]"},
  };
}

json truncation(std::uint32_t max_length, std::uint32_t stride = 0) {
  return {
      {"direction", "Right"},
      {"max_length", max_length},
      {"strategy", "LongestFirst"},
      {"stride", stride},
  };
}

std::vector<tokenizers_cpp::Offset> offsets(
    std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

void test_fixed_right_padding_single_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_fixed_right.json",
      wordlevel_tokenizer_json(fixed_padding(5)));

  const auto output = tokenizer.encode("one two", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 0, 0, 0}));
  assert((output.tokens == std::vector<std::string>{"one", "two", "[PAD]", "[PAD]", "[PAD]"}));
  assert((output.offsets == offsets({{0, 3}, {4, 7}, {0, 0}, {0, 0}, {0, 0}})));
  assert((output.word_ids == word_ids({0U, 1U, std::nullopt, std::nullopt, std::nullopt})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
}

void test_fixed_left_padding_single_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_fixed_left.json",
      wordlevel_tokenizer_json(fixed_padding(4, "Left", nullptr, 7)));

  const auto output = tokenizer.encode("one two", false);
  assert((output.ids == std::vector<std::uint32_t>{0, 0, 1, 2}));
  assert((output.tokens == std::vector<std::string>{"[PAD]", "[PAD]", "one", "two"}));
  assert((output.offsets == offsets({{0, 0}, {0, 0}, {0, 3}, {4, 7}})));
  assert((output.word_ids == word_ids({std::nullopt, std::nullopt, 0U, 1U})));
  assert((output.type_ids == std::vector<std::uint32_t>{7, 7, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 1, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{0, 0, 1, 1}));
}

void test_fixed_padding_to_multiple() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_fixed_multiple.json",
      wordlevel_tokenizer_json(fixed_padding(5, "Right", 4)));

  const auto output = tokenizer.encode("one two", false);
  assert(output.ids.size() == 8);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0, 0, 0}));
}

void test_padding_applies_to_single_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_single_overflow.json",
      wordlevel_tokenizer_json(fixed_padding(5), truncation(3)));

  const auto output = tokenizer.encode("one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0}));
  assert(output.overflowing.size() == 1);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{4, 5, 0, 0, 0}));
  assert((output.overflowing[0].special_tokens_mask == std::vector<std::uint32_t>{0, 0, 1, 1, 1}));
  assert((output.overflowing[0].attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
}

void test_batch_longest_padding_targets_main_length_and_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_longest_overflow.json",
      wordlevel_tokenizer_json(batch_longest_padding(), truncation(3)));

  const auto output = tokenizer.encode("one two three four five", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert(output.overflowing.size() == 1);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{4, 5, 0}));
  assert((output.overflowing[0].attention_mask == std::vector<std::uint32_t>{1, 1, 0}));
}

void test_fixed_padding_pair_sequence_with_special_tokens() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_pair_specials.json",
      wordlevel_tokenizer_json(
          fixed_padding(7),
          nullptr,
          {{"type", "BertProcessing"},
           {"sep", {"[SEP]", 102}},
           {"cls", {"[CLS]", 101}}}));

  const auto output = tokenizer.encode_pair("one", "two", true);
  assert((output.ids == std::vector<std::uint32_t>{101, 1, 102, 2, 102, 0, 0}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "one",
                               "[SEP]",
                               "two",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0}));
}

void test_padding_applies_to_pair_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_pair_overflow.json",
      wordlevel_tokenizer_json(fixed_padding(6), truncation(5)));

  const auto output = tokenizer.encode_pair("one two three", "one two three", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 1, 2, 3, 0}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 1, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0}));
  assert(output.overflowing.size() == 1);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{3, 1, 2, 3, 0, 0}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 1, 1, 1, 0, 0}));
  assert((output.overflowing[0].attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));
}

void test_fixed_padding_pre_tokenized_pair_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_pre_tokenized_pair_overflow.json",
      wordlevel_tokenizer_json(fixed_padding(5), truncation(4)));

  const auto output = tokenizer.encode_pair(
      std::vector<std::string>{"one", "two", "three"},
      std::vector<std::string>{"one", "two", "three"},
      false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 1, 2, 0}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0}));
  assert((output.offsets == offsets({
                               {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 0}})));
  assert((output.word_ids == word_ids({0U, 1U, 0U, 1U, std::nullopt})));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0}));

  assert(output.overflowing.size() == 3);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{3, 1, 2, 0, 0}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 1, 1, 0, 0}));
  assert((output.overflowing[0].word_ids == word_ids({
                                                 2U,
                                                 0U,
                                                 1U,
                                                 std::nullopt,
                                                 std::nullopt,
                                             })));
  assert((output.overflowing[0].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 1, 0, 0}));
  assert((output.overflowing[1].ids == std::vector<std::uint32_t>{3, 3, 0, 0, 0}));
  assert((output.overflowing[1].type_ids == std::vector<std::uint32_t>{0, 1, 0, 0, 0}));
  assert((output.overflowing[1].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
  assert((output.overflowing[2].ids == std::vector<std::uint32_t>{1, 2, 3, 0, 0}));
  assert((output.overflowing[2].type_ids == std::vector<std::uint32_t>{0, 0, 1, 0, 0}));
  assert((output.overflowing[2].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 1, 0, 0}));
}

void test_invalid_padding_direction_rejected() {
  try {
    (void)load_tokenizer(
        "tokenizers_cpp_padding_invalid_direction.json",
        wordlevel_tokenizer_json({
            {"strategy", "BatchLongest"},
            {"direction", "Center"},
            {"pad_to_multiple_of", nullptr},
            {"pad_id", 0},
            {"pad_type_id", 0},
            {"pad_token", "[PAD]"},
        }));
    assert(false && "invalid padding direction should fail");
  } catch (const std::runtime_error & error) {
    assert(std::string(error.what()).find("direction") != std::string::npos);
  }
}

void test_batch_longest_padding_across_single_encodings() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_single.json",
      wordlevel_tokenizer_json(batch_longest_padding()));

  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"one two", "one two three four"},
      false);
  assert(outputs.size() == 2);
  assert((outputs[0].ids == std::vector<std::uint32_t>{1, 2, 0, 0}));
  assert((outputs[0].tokens == std::vector<std::string>{"one", "two", "[PAD]", "[PAD]"}));
  assert((outputs[0].attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0}));
  assert((outputs[0].special_tokens_mask == std::vector<std::uint32_t>{0, 0, 1, 1}));
  assert((outputs[1].ids == std::vector<std::uint32_t>{1, 2, 3, 4}));
  assert((outputs[1].attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_batch_longest_padding_to_multiple() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_multiple.json",
      wordlevel_tokenizer_json(batch_longest_padding(4)));

  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"one two", "one two three four five"},
      false);
  assert(outputs.size() == 2);
  assert(outputs[0].ids.size() == 8);
  assert(outputs[1].ids.size() == 8);
  assert((outputs[0].ids == std::vector<std::uint32_t>{1, 2, 0, 0, 0, 0, 0, 0}));
  assert((outputs[1].ids == std::vector<std::uint32_t>{1, 2, 3, 4, 5, 0, 0, 0}));
  assert((outputs[1].attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

void test_batch_longest_padding_across_pair_encodings() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_pair.json",
      wordlevel_tokenizer_json(
          batch_longest_padding(),
          nullptr,
          {{"type", "BertProcessing"},
           {"sep", {"[SEP]", 102}},
           {"cls", {"[CLS]", 101}}}));

  const auto outputs = tokenizer.encode_batch_pairs(
      {{"one", "two"}, {"one two", "three four five"}},
      true);
  assert(outputs.size() == 2);
  assert((outputs[0].ids == std::vector<std::uint32_t>{101, 1, 102, 2, 102, 0, 0, 0}));
  assert((outputs[0].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert((outputs[0].special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((outputs[0].attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
  assert((outputs[1].ids == std::vector<std::uint32_t>{101, 1, 2, 102, 3, 4, 5, 102}));
  assert((outputs[1].type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 1, 1}));
}

void test_batch_longest_padding_applies_to_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_overflow.json",
      wordlevel_tokenizer_json(batch_longest_padding(), truncation(4)));

  const auto outputs =
      tokenizer.encode_batch(
          std::vector<std::string>{"one two three four five", "one two three"},
          false);
  assert(outputs.size() == 2);
  assert((outputs[0].ids == std::vector<std::uint32_t>{1, 2, 3, 4}));
  assert(outputs[0].overflowing.size() == 1);
  assert((outputs[0].overflowing[0].ids == std::vector<std::uint32_t>{5, 0, 0, 0}));
  assert((outputs[0].overflowing[0].attention_mask == std::vector<std::uint32_t>{1, 0, 0, 0}));
  assert((outputs[1].ids == std::vector<std::uint32_t>{1, 2, 3, 0}));
  assert((outputs[1].attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0}));
}

void test_batch_longest_padding_pre_tokenized_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_pre_tokenized_overflow.json",
      wordlevel_tokenizer_json(batch_longest_padding(), truncation(4)));

  const std::vector<std::vector<std::string>> inputs = {
      {"one", "two"},
      {"one", "two", "three", "four", "five"},
  };
  const auto outputs = tokenizer.encode_batch(inputs, false);
  assert(outputs.size() == 2);

  assert((outputs[0].ids == std::vector<std::uint32_t>{1, 2, 0, 0}));
  assert((outputs[0].offsets == offsets({{0, 3}, {0, 3}, {0, 0}, {0, 0}})));
  assert((outputs[0].word_ids == word_ids({0U, 1U, std::nullopt, std::nullopt})));
  assert((outputs[0].attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0}));

  assert((outputs[1].ids == std::vector<std::uint32_t>{1, 2, 3, 4}));
  assert((outputs[1].offsets == offsets({{0, 3}, {0, 3}, {0, 5}, {0, 4}})));
  assert((outputs[1].word_ids == word_ids({0U, 1U, 2U, 3U})));
  assert(outputs[1].overflowing.size() == 1);
  assert((outputs[1].overflowing[0].ids == std::vector<std::uint32_t>{5, 0, 0, 0}));
  assert((outputs[1].overflowing[0].offsets == offsets(
                                             {{0, 4}, {0, 0}, {0, 0}, {0, 0}})));
  assert((outputs[1].overflowing[0].word_ids == word_ids(
                                                 {
                                                     4U,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                 })));
  assert((outputs[1].overflowing[0].attention_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0}));
}

void test_batch_longest_padding_pre_tokenized_pair_overflowing() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_padding_batch_pre_tokenized_pair_overflow.json",
      wordlevel_tokenizer_json(batch_longest_padding(), truncation(4)));

  using PretokPair = std::pair<std::vector<std::string>, std::vector<std::string>>;
  const std::vector<PretokPair> inputs = {
      {{"one"}, {"two"}},
      {{"one", "two", "three"}, {"one", "two", "three"}},
  };
  const auto outputs = tokenizer.encode_batch_pairs(inputs, false);
  assert(outputs.size() == 2);

  assert((outputs[0].ids == std::vector<std::uint32_t>{1, 2, 0, 0}));
  assert((outputs[0].type_ids == std::vector<std::uint32_t>{0, 1, 0, 0}));
  assert((outputs[0].word_ids == word_ids({0U, 0U, std::nullopt, std::nullopt})));
  assert((outputs[0].attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0}));

  assert((outputs[1].ids == std::vector<std::uint32_t>{1, 2, 1, 2}));
  assert((outputs[1].type_ids == std::vector<std::uint32_t>{0, 0, 1, 1}));
  assert((outputs[1].word_ids == word_ids({0U, 1U, 0U, 1U})));
  assert(outputs[1].overflowing.size() == 3);
  assert((outputs[1].overflowing[0].ids == std::vector<std::uint32_t>{3, 1, 2, 0}));
  assert((outputs[1].overflowing[0].type_ids == std::vector<std::uint32_t>{0, 1, 1, 0}));
  assert((outputs[1].overflowing[0].word_ids == word_ids(
                                                 {2U, 0U, 1U, std::nullopt})));
  assert((outputs[1].overflowing[0].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 1, 0}));
  assert((outputs[1].overflowing[1].ids == std::vector<std::uint32_t>{3, 3, 0, 0}));
  assert((outputs[1].overflowing[1].type_ids == std::vector<std::uint32_t>{0, 1, 0, 0}));
  assert((outputs[1].overflowing[1].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 0, 0}));
  assert((outputs[1].overflowing[2].ids == std::vector<std::uint32_t>{1, 2, 3, 0}));
  assert((outputs[1].overflowing[2].type_ids == std::vector<std::uint32_t>{0, 0, 1, 0}));
  assert((outputs[1].overflowing[2].attention_mask ==
          std::vector<std::uint32_t>{1, 1, 1, 0}));
}

}  // namespace

int main() {
  test_fixed_right_padding_single_sequence();
  test_fixed_left_padding_single_sequence();
  test_fixed_padding_to_multiple();
  test_padding_applies_to_single_overflowing();
  test_batch_longest_padding_targets_main_length_and_overflowing();
  test_fixed_padding_pair_sequence_with_special_tokens();
  test_padding_applies_to_pair_overflowing();
  test_fixed_padding_pre_tokenized_pair_overflowing();
  test_invalid_padding_direction_rejected();
  test_batch_longest_padding_across_single_encodings();
  test_batch_longest_padding_to_multiple();
  test_batch_longest_padding_across_pair_encodings();
  test_batch_longest_padding_applies_to_overflowing();
  test_batch_longest_padding_pre_tokenized_overflowing();
  test_batch_longest_padding_pre_tokenized_pair_overflowing();
  return 0;
}
