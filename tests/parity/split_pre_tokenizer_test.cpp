#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
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

std::vector<tokenizers_cpp::Offset> offsets(
    std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

tokenizers_cpp::Tokenizer load_wordlevel_split_tokenizer(
    const std::string & name,
    const json & vocab,
    const json & split) {
  const json tokenizer_json = {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", split},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "WordLevel"},
           {"unk_token", "[UNK]"},
           {"vocab", vocab},
       }},
  };

  const auto path = write_temp_tokenizer_json(name, tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_split_string_removed() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_split_string_removed.json",
      {
          {"[UNK]", 0},
          {"Hey,", 1},
          {"man!", 2},
      },
      {
          {"type", "Split"},
          {"pattern", {{"String", " "}}},
          {"behavior", "Removed"},
          {"invert", false},
      });

  const auto output = tokenizer.encode("Hey, man!", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"Hey,", "man!"}));
  assert((output.offsets == offsets({{0, 4}, {5, 9}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_split_string_invert_removed() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_split_string_invert_removed.json",
      {
          {"[UNK]", 0},
          {"Hello", 1},
      },
      {
          {"type", "Split"},
          {"pattern", {{"String", "Hello"}}},
          {"behavior", "Removed"},
          {"invert", true},
      });

  const auto output = tokenizer.encode("Hello Hello Hello", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 1, 1}));
  assert((output.tokens == std::vector<std::string>{"Hello", "Hello", "Hello"}));
  assert((output.offsets == offsets({{0, 5}, {6, 11}, {12, 17}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_split_string_delimiter_behaviors() {
  const json vocab = {
      {"[UNK]", 0},
      {"The", 1},
      {"final", 2},
      {"countdown", 3},
      {"-", 4},
      {"The-", 5},
      {"final-", 6},
      {"-final", 7},
      {"-countdown", 8},
      {"--", 9},
  };

  const auto load = [&](const std::string & behavior) {
    return load_wordlevel_split_tokenizer(
        "tokenizers_cpp_split_string_" + behavior + ".json",
        vocab,
        {
            {"type", "Split"},
            {"pattern", {{"String", "-"}}},
            {"behavior", behavior},
            {"invert", false},
        });
  };

  {
    const auto output = load("Removed").encode("The-final--countdown", false);
    assert((output.tokens == std::vector<std::string>{"The", "final", "countdown"}));
    assert((output.offsets == offsets({{0, 3}, {4, 9}, {11, 20}})));
    assert(output.word_ids == word_ids({0U, 1U, 2U}));
  }
  {
    const auto output = load("Isolated").encode("The-final--countdown", false);
    assert((output.tokens == std::vector<std::string>{
                                 "The", "-", "final", "-", "-", "countdown"}));
    assert((output.offsets == offsets({{0, 3}, {3, 4}, {4, 9}, {9, 10}, {10, 11}, {11, 20}})));
    assert(output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U, 5U}));
  }
  {
    const auto output = load("MergedWithPrevious").encode("The-final--countdown", false);
    assert((output.tokens == std::vector<std::string>{
                                 "The-", "final-", "-", "countdown"}));
    assert((output.offsets == offsets({{0, 4}, {4, 10}, {10, 11}, {11, 20}})));
    assert(output.word_ids == word_ids({0U, 1U, 2U, 3U}));
  }
  {
    const auto output = load("MergedWithNext").encode("The-final--countdown", false);
    assert((output.tokens == std::vector<std::string>{
                                 "The", "-final", "-", "-countdown"}));
    assert((output.offsets == offsets({{0, 3}, {3, 9}, {9, 10}, {10, 20}})));
    assert(output.word_ids == word_ids({0U, 1U, 2U, 3U}));
  }
  {
    const auto output = load("Contiguous").encode("The-final--countdown", false);
    assert((output.tokens == std::vector<std::string>{
                                 "The", "-", "final", "--", "countdown"}));
    assert((output.offsets == offsets({{0, 3}, {3, 4}, {4, 9}, {9, 11}, {11, 20}})));
    assert(output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U}));
  }
}

void test_split_icu_regex_punctuation_isolated() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_split_icu_regex_punctuation_isolated.json",
      {
          {"[UNK]", 0},
          {"hello", 1},
          {"\xF0\x90\x84\x80", 2},
          {"world", 3},
      },
      {
          {"type", "Split"},
          {"pattern", {{"Regex", "\\p{P}+"}}},
          {"behavior", "Isolated"},
          {"invert", false},
      });

  const auto output = tokenizer.encode("hello" "\xF0\x90\x84\x80" "world", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               "\xF0\x90\x84\x80",
                               "world",
                           }));
  assert((output.offsets == offsets({{0, 5}, {5, 9}, {9, 14}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_digits_pre_tokenizer_individual_digits() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_digits_individual.json",
      {
          {"[UNK]", 0},
          {"Hey ", 1},
          {"1", 2},
          {"2", 3},
          {"3", 4},
          {" friend!", 5},
      },
      {
          {"type", "Digits"},
          {"individual_digits", true},
      });

  const auto output = tokenizer.encode("Hey 123 friend!", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 4, 5}));
  assert((output.tokens == std::vector<std::string>{
                               "Hey ",
                               "1",
                               "2",
                               "3",
                               " friend!",
                           }));
  assert((output.offsets == offsets({{0, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 15}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
}

void test_digits_pre_tokenizer_contiguous_digits() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_digits_contiguous.json",
      {
          {"[UNK]", 0},
          {"Hey ", 1},
          {"123", 2},
          {" friend!", 3},
      },
      {
          {"type", "Digits"},
          {"individual_digits", false},
      });

  const auto output = tokenizer.encode("Hey 123 friend!", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert((output.tokens == std::vector<std::string>{"Hey ", "123", " friend!"}));
  assert((output.offsets == offsets({{0, 4}, {4, 7}, {7, 15}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_whitespace_then_digits_sequence() {
  const auto tokenizer = load_wordlevel_split_tokenizer(
      "tokenizers_cpp_whitespace_digits_sequence.json",
      {
          {"[UNK]", 0},
          {"Call", 1},
          {"9", 2},
          {"1", 3},
          {"!", 4},
      },
      {
          {"type", "Sequence"},
          {"pretokenizers",
           json::array({
               {{"type", "Whitespace"}},
               {{"type", "Digits"}, {"individual_digits", true}},
           })},
      });

  const auto output = tokenizer.encode("Call 911!", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 3, 4}));
  assert((output.tokens == std::vector<std::string>{"Call", "9", "1", "1", "!"}));
  assert((output.offsets == offsets({{0, 4}, {5, 6}, {6, 7}, {7, 8}, {8, 9}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
}

}  // namespace

int main() {
  test_split_string_removed();
  test_split_string_invert_removed();
  test_split_string_delimiter_behaviors();
  test_split_icu_regex_punctuation_isolated();
  test_digits_pre_tokenizer_individual_digits();
  test_digits_pre_tokenizer_contiguous_digits();
  test_whitespace_then_digits_sequence();
  return 0;
}
