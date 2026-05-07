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

tokenizers_cpp::Tokenizer load_unigram_normalizer_tokenizer(
    const std::string & name,
    const json & normalizer,
    const json & vocab) {
  const json tokenizer_json = {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", normalizer},
      {"pre_tokenizer",
       {
           {"type", "Sequence"},
           {"pretokenizers", json::array({{{"type", "WhitespaceSplit"}}})},
       }},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "Unigram"},
           {"unk_id", 0},
           {"byte_fallback", false},
           {"vocab", vocab},
       }},
  };

  const auto path = write_temp_tokenizer_json(name, tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

json unigram_vocab(std::initializer_list<std::string> tokens) {
  json vocab = json::array({json::array({"<unk>", 0.0})});
  double score = 1.0;
  for (const auto & token : tokens) {
    vocab.push_back(json::array({token, score}));
    score += 1.0;
  }
  return vocab;
}

void test_replace_string_normalizer() {
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_replace_string_normalizer.json",
      {
          {"type", "Replace"},
          {"pattern", {{"String", "_"}}},
          {"content", " "},
      },
      unigram_vocab({"hello", "world"}));

  const auto output = tokenizer.encode("hello_world", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"hello", "world"}));
  assert((output.offsets == offsets({{0, 5}, {6, 11}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_replace_regex_whitespace_normalizer() {
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_replace_regex_whitespace_normalizer.json",
      {
          {"type", "Replace"},
          {"pattern", {{"Regex", "\\s+"}}},
          {"content", " "},
      },
      unigram_vocab({"This", "is", "a", "test"}));

  const auto output = tokenizer.encode("This     is   a         test", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3, 4}));
  assert((output.tokens == std::vector<std::string>{"This", "is", "a", "test"}));
  assert((output.offsets == offsets({{0, 4}, {9, 11}, {14, 15}, {24, 28}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U, 3U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_replace_regex_unicode_punctuation_normalizer() {
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_replace_regex_unicode_punctuation_normalizer.json",
      {
          {"type", "Replace"},
          {"pattern", {{"Regex", "\\p{P}+"}}},
          {"content", " "},
      },
      unigram_vocab({"hello", "world"}));

  const auto output = tokenizer.encode("hello" "\xF0\x90\x84\x80" "world", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"hello", "world"}));
  assert((output.offsets == offsets({{0, 5}, {9, 14}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_strip_then_nfc_normalizer() {
  const auto cafe = std::string("caf") + "\xC3\xA9";
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_strip_then_nfc_normalizer.json",
      {
          {"type", "Sequence"},
          {"normalizers",
           json::array({
               {
                   {"type", "Strip"},
                   {"strip_left", true},
                   {"strip_right", true},
               },
               {
                   {"type", "NFC"},
               },
           })},
      },
      unigram_vocab({cafe, "hi"}));

  const auto output = tokenizer.encode("  cafe\xCC\x81  hi  ", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{cafe, "hi"}));
  assert((output.offsets == offsets({{2, 8}, {10, 12}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_nfkc_normalizer() {
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_nfkc_normalizer.json",
      {
          {"type", "NFKC"},
      },
      unigram_vocab({"fi", "one"}));

  const auto output = tokenizer.encode("\xEF\xAC\x81 one", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"fi", "one"}));
  assert((output.offsets == offsets({{0, 3}, {4, 7}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_nmt_normalizer() {
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_nmt_normalizer.json",
      {
          {"type", "Nmt"},
      },
      unigram_vocab({"A", "B", "C"}));

  const auto output = tokenizer.encode(
      std::string("A\t") + "\xE2\x80\x8B" + "B\x01 C",
      false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert((output.tokens == std::vector<std::string>{"A", "B", "C"}));
  assert((output.offsets == offsets({{0, 1}, {5, 6}, {8, 9}})));
  assert(output.word_ids == word_ids({0U, 1U, 2U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_prepend_normalizer() {
  const auto marker = std::string("\xE2\x96\x81");
  const auto tokenizer = load_unigram_normalizer_tokenizer(
      "tokenizers_cpp_prepend_normalizer.json",
      {
          {"type", "Prepend"},
          {"prepend", marker},
      },
      unigram_vocab({marker + "Hello"}));

  const auto output = tokenizer.encode("Hello", false);
  assert((output.ids == std::vector<std::uint32_t>{1}));
  assert((output.tokens == std::vector<std::string>{marker + "Hello"}));
  assert((output.offsets == offsets({{0, 5}})));
  assert(output.word_ids == word_ids({0U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1}));

  const auto empty = tokenizer.encode("", false);
  assert(empty.ids.empty());
  assert(empty.tokens.empty());
  assert(empty.offsets.empty());
}

}  // namespace

int main() {
  test_replace_string_normalizer();
  test_replace_regex_whitespace_normalizer();
  test_replace_regex_unicode_punctuation_normalizer();
  test_strip_then_nfc_normalizer();
  test_nfkc_normalizer();
  test_nmt_normalizer();
  test_prepend_normalizer();
  return 0;
}
