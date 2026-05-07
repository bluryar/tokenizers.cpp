#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef TOKENIZERS_CPP_HF_TEST_DATA_DIR
#error "TOKENIZERS_CPP_HF_TEST_DATA_DIR must be defined"
#endif

namespace {

using json = nlohmann::json;

std::string bl_space() {
  return "\xC4\xA0";
}

std::string bl_cat_prefix() {
  return "\xC3\xB0\xC5\x81\xC4\xBA";
}

std::string bl_arrow_first() {
  return "\xC3\xA2";
}

std::string bl_arrow_second() {
  return "\xC5\x83";
}

std::string bl_arrow_third() {
  return "\xC2\xA2";
}

std::string bl_cat_tail() {
  return "\xC2\xBA";
}

std::string cat_face() {
  return "\xF0\x9F\x98\xBA";
}

std::string arrow() {
  return "\xE2\xAD\xA2";
}

std::string aegean_word_separator() {
  return "\xF0\x90\x84\x80";
}

json read_json(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open JSON fixture");
  json value;
  input >> value;
  return value;
}

json read_bpe_merges(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open BPE merges fixture");

  json merges = json::array();
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream stream(line);
    std::string left;
    std::string right;
    stream >> left >> right;
    if (!left.empty() && !right.empty()) {
      merges.push_back(json::array({left, right}));
    }
  }
  return merges;
}

std::filesystem::path write_temp_tokenizer_json(const std::string & name, const json & value) {
  static std::uint64_t counter = 0;
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
      (name + "." + std::to_string(stamp) + "." + std::to_string(++counter));
  std::ofstream output(path);
  output << value;
  return path;
}

tokenizers_cpp::Tokenizer load_byte_level_bpe(
    const std::filesystem::path & data_dir,
    bool add_prefix_space,
    bool trim_offsets,
    json added_tokens = json::array()) {
  const auto tokenizer_json = json{
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", std::move(added_tokens)},
      {"normalizer", nullptr},
      {"pre_tokenizer",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", add_prefix_space},
           {"trim_offsets", true},
           {"use_regex", true},
       }},
      {"post_processor",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", true},
           {"trim_offsets", trim_offsets},
           {"use_regex", true},
       }},
      {"decoder",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", true},
           {"trim_offsets", true},
           {"use_regex", true},
       }},
      {"model",
       {
           {"type", "BPE"},
           {"dropout", nullptr},
           {"unk_token", nullptr},
           {"continuing_subword_prefix", nullptr},
           {"end_of_word_suffix", nullptr},
           {"fuse_unk", false},
           {"byte_fallback", false},
           {"ignore_merges", false},
           {"vocab", read_json(data_dir / "gpt2-vocab.json")},
           {"merges", read_bpe_merges(data_dir / "gpt2-merges.txt")},
       }},
  };

  const auto path = write_temp_tokenizer_json(
      add_prefix_space ? "tokenizers_cpp_byte_level_prefix.json"
                       : "tokenizers_cpp_byte_level_no_prefix.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_byte_level_bpe_pipeline(
    const std::filesystem::path & data_dir,
    const std::string & name,
    json normalizer,
    json pre_tokenizer) {
  const auto tokenizer_json = json{
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", std::move(normalizer)},
      {"pre_tokenizer", std::move(pre_tokenizer)},
      {"post_processor",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", true},
           {"trim_offsets", false},
           {"use_regex", true},
       }},
      {"decoder",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", true},
           {"trim_offsets", true},
           {"use_regex", true},
       }},
      {"model",
       {
           {"type", "BPE"},
           {"dropout", nullptr},
           {"unk_token", nullptr},
           {"continuing_subword_prefix", nullptr},
           {"end_of_word_suffix", nullptr},
           {"fuse_unk", false},
           {"byte_fallback", false},
           {"ignore_merges", false},
           {"vocab", read_json(data_dir / "gpt2-vocab.json")},
           {"merges", read_bpe_merges(data_dir / "gpt2-merges.txt")},
       }},
  };

  const auto path = write_temp_tokenizer_json(name, tokenizer_json);
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

void check_common_encoding_vectors(
    const tokenizers_cpp::Tokenizer & tokenizer,
    const tokenizers_cpp::Encoding & output,
    const std::vector<std::uint32_t> & expected_special_tokens_mask,
    const std::vector<std::optional<std::uint32_t>> & expected_word_ids) {
  assert(output.ids.size() == output.tokens.size());
  assert(output.type_ids == std::vector<std::uint32_t>(output.ids.size(), 0));
  assert(output.attention_mask == std::vector<std::uint32_t>(output.ids.size(), 1));
  assert(output.special_tokens_mask == expected_special_tokens_mask);
  assert(output.word_ids == expected_word_ids);

  for (std::size_t index = 0; index < output.tokens.size(); ++index) {
    const auto lookup_token =
        output.tokens[index].find("<mask>") == std::string::npos
        ? output.tokens[index]
        : std::string("<mask>");
    assert(output.ids[index] == tokenizer.token_to_id(lookup_token).value());
  }
}

void check_offset_slice(
    const std::string & input,
    const tokenizers_cpp::Encoding & output,
    std::size_t index,
    const std::string & expected) {
  const auto offset = output.offsets.at(index);
  assert(input.substr(offset.start, offset.end - offset.start) == expected);
}

json mask_token(bool lstrip, bool rstrip) {
  return json::array({
      {
          {"id", 0},
          {"content", "<mask>"},
          {"single_word", false},
          {"lstrip", lstrip},
          {"rstrip", rstrip},
          {"normalized", false},
          {"special", true},
      },
  });
}

void test_byte_level_basic(const std::filesystem::path & data_dir) {
  const std::string input = "Hello there, how are you?";
  const auto tokenizer = load_byte_level_bpe(data_dir, true, false);
  const auto output = tokenizer.encode(input, false);

  assert((output.tokens == std::vector<std::string>{
                               bl_space() + "Hello",
                               bl_space() + "there",
                               ",",
                               bl_space() + "how",
                               bl_space() + "are",
                               bl_space() + "you",
                               "?"}));
  assert((output.offsets == offsets({
                                {0, 5},
                                {5, 11},
                                {11, 12},
                                {12, 16},
                                {16, 20},
                                {20, 24},
                                {24, 25},
                            })));
  check_common_encoding_vectors(
      tokenizer,
      output,
      {0, 0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 4, 5, 6}));

  const auto trimming_tokenizer = load_byte_level_bpe(data_dir, true, true);
  const auto trimmed = trimming_tokenizer.encode(input, false);
  assert((trimmed.offsets == offsets({
                                 {0, 5},
                                 {6, 11},
                                 {11, 12},
                                 {13, 16},
                                 {17, 20},
                                 {21, 24},
                                 {24, 25},
                             })));
  check_common_encoding_vectors(
      trimming_tokenizer,
      trimmed,
      {0, 0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 4, 5, 6}));
}

void test_byte_level_unicode(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_byte_level_bpe(data_dir, true, false);
  const std::string input = std::string("i") + arrow() + "j";
  const auto output = tokenizer.encode(input, false);

  assert((output.tokens == std::vector<std::string>{
                               bl_space() + "i",
                               bl_arrow_first(),
                               bl_arrow_second(),
                               bl_arrow_third(),
                               "j"}));
  check_offset_slice(input, output, 1, arrow());
  check_offset_slice(input, output, 2, arrow());
  check_offset_slice(input, output, 3, arrow());
  check_common_encoding_vectors(
      tokenizer,
      output,
      {0, 0, 0, 0, 0},
      word_ids({0, 1, 1, 1, 2}));
}

void test_replace_regex_before_byte_level_bpe(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_byte_level_bpe_pipeline(
      data_dir,
      "tokenizers_cpp_replace_regex_byte_level_bpe.json",
      {
          {"type", "Replace"},
          {"pattern", {{"Regex", "\\p{P}+"}}},
          {"content", " "},
      },
      {
          {"type", "ByteLevel"},
          {"add_prefix_space", false},
          {"trim_offsets", true},
          {"use_regex", true},
      });

  const auto input = std::string("hello") + aegean_word_separator() + "world";
  const auto output = tokenizer.encode(input, false);
  assert((output.ids == std::vector<std::uint32_t>{31373, 995}));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               bl_space() + "world"}));
  assert((output.offsets == offsets({{0, 5}, {5, 14}})));
  assert(output.word_ids == word_ids({0U, 1U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(output.ids, false) == "hello world");
}

void test_split_regex_before_byte_level_bpe(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_byte_level_bpe_pipeline(
      data_dir,
      "tokenizers_cpp_split_regex_byte_level_bpe.json",
      nullptr,
      {
          {"type", "Sequence"},
          {"pretokenizers",
           json::array({
               {
                   {"type", "Split"},
                   {"pattern", {{"Regex", "\\p{P}+"}}},
                   {"behavior", "Isolated"},
                   {"invert", false},
               },
               {
                   {"type", "ByteLevel"},
                   {"add_prefix_space", false},
                   {"trim_offsets", true},
                   {"use_regex", false},
               },
           })},
      });

  const auto input = std::string("hello") + aegean_word_separator() + "world";
  const auto output = tokenizer.encode(input, false);
  assert((output.ids == std::vector<std::uint32_t>{
                            31373, 172, 238, 226, 222, 6894}));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               "\xC3\xB0",
                               "\xC4\xB2",
                               "\xC4\xA6",
                               "\xC4\xA2",
                               "world"}));
  assert((output.offsets == offsets({
                              {0, 5},
                              {5, 9},
                              {5, 9},
                              {5, 9},
                              {5, 9},
                              {9, 14},
                          })));
  assert(output.word_ids == word_ids({0U, 1U, 1U, 1U, 1U, 2U}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert((
      output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert((
      output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == input);
}

void test_byte_level_double_sequence(const std::filesystem::path & data_dir) {
  const std::string input_a = "My name is Anthony";
  const std::string input_b = "What is my name?";

  const auto tokenizer = load_byte_level_bpe(data_dir, true, false);
  const auto output = tokenizer.encode_pair(input_a, input_b, false);
  const std::vector<std::uint32_t> expected_ids = {
      2011, 1438, 318, 9953, 1867, 318, 616, 1438, 30};
  const std::vector<std::string> expected_tokens = {
      bl_space() + "My",
      bl_space() + "name",
      bl_space() + "is",
      bl_space() + "Anthony",
      bl_space() + "What",
      bl_space() + "is",
      bl_space() + "my",
      bl_space() + "name",
      "?"};
  assert(output.ids == expected_ids);
  assert(output.tokens == expected_tokens);
  assert((output.offsets == offsets({
                                {0, 2},
                                {2, 7},
                                {7, 10},
                                {10, 18},
                                {0, 4},
                                {4, 7},
                                {7, 10},
                                {10, 15},
                                {15, 16},
                            })));
  assert((output.word_ids == word_ids({0, 1, 2, 3, 0, 1, 2, 3, 4})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 1, 1, 1}));
  assert(output.attention_mask == std::vector<std::uint32_t>(output.ids.size(), 1));
  assert(output.special_tokens_mask == std::vector<std::uint32_t>(output.ids.size(), 0));

  const auto trimming_tokenizer = load_byte_level_bpe(data_dir, true, true);
  const auto trimmed = trimming_tokenizer.encode_pair(input_a, input_b, false);
  assert(trimmed.ids == expected_ids);
  assert(trimmed.tokens == expected_tokens);
  assert((trimmed.offsets == offsets({
                                 {0, 2},
                                 {3, 7},
                                 {8, 10},
                                 {11, 18},
                                 {0, 4},
                                 {5, 7},
                                 {8, 10},
                                 {11, 15},
                                 {15, 16},
                             })));
  assert((trimmed.word_ids == word_ids({0, 1, 2, 3, 0, 1, 2, 3, 4})));
  assert((trimmed.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 1, 1, 1}));
  assert(trimmed.attention_mask == std::vector<std::uint32_t>(trimmed.ids.size(), 1));
  assert(trimmed.special_tokens_mask == std::vector<std::uint32_t>(trimmed.ids.size(), 0));
}

void test_byte_level_pre_tokenized_sequence(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_byte_level_bpe(data_dir, true, false);
  const auto output = tokenizer.encode(
      std::vector<std::string>{"My", "name", "is", "Anthonino"},
      false);

  assert((output.ids == std::vector<std::uint32_t>{2011, 1438, 318, 8451, 261, 2879}));
  assert((output.tokens == std::vector<std::string>{
                               bl_space() + "My",
                               bl_space() + "name",
                               bl_space() + "is",
                               bl_space() + "Anth",
                               "on",
                               "ino"}));
  assert((output.word_ids == word_ids({0, 1, 2, 3, 3, 3})));
  assert((output.offsets == offsets({
                                {0, 2},
                                {0, 4},
                                {0, 2},
                                {0, 4},
                                {4, 6},
                                {6, 9},
                            })));
  check_common_encoding_vectors(
      tokenizer,
      output,
      {0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 3, 3}));
}

void test_added_token_lstrip_byte_level(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_byte_level_bpe(data_dir, true, false, mask_token(true, false));
  const std::string input = std::string("I saw a <mask> ") + cat_face();
  const auto output = tokenizer.encode(input, false);

  assert((output.tokens == std::vector<std::string>{
                               bl_space() + "I",
                               bl_space() + "saw",
                               bl_space() + "a",
                               " <mask>",
                               bl_space() + bl_cat_prefix(),
                               bl_cat_tail()}));
  assert((output.offsets == offsets({
                                {0, 1},
                                {1, 5},
                                {5, 7},
                                {7, 14},
                                {14, 19},
                                {15, 19},
                            })));
  check_common_encoding_vectors(
      tokenizer,
      output,
      {0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 4, 4}));
}

void test_added_token_rstrip_byte_level(const std::filesystem::path & data_dir) {
  const std::string input = std::string("I saw a <mask> ") + cat_face();
  const auto tokenizer = load_byte_level_bpe(data_dir, false, false, mask_token(false, true));
  const auto output = tokenizer.encode(input, false);

  assert((output.tokens == std::vector<std::string>{
                               "I",
                               bl_space() + "saw",
                               bl_space() + "a",
                               bl_space(),
                               "<mask> ",
                               bl_cat_prefix(),
                               bl_cat_tail()}));
  assert((output.offsets == offsets({
                                {0, 1},
                                {1, 5},
                                {5, 7},
                                {7, 8},
                                {8, 15},
                                {15, 19},
                                {15, 19},
                            })));
  check_common_encoding_vectors(
      tokenizer,
      output,
      {0, 0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 4, 5, 5}));

  const auto prefix_tokenizer =
      load_byte_level_bpe(data_dir, true, false, mask_token(false, true));
  const auto prefix_output = prefix_tokenizer.encode(input, false);
  assert((prefix_output.tokens == std::vector<std::string>{
                                      bl_space() + "I",
                                      bl_space() + "saw",
                                      bl_space() + "a",
                                      bl_space(),
                                      "<mask> ",
                                      bl_space() + bl_cat_prefix(),
                                      bl_cat_tail()}));
  check_common_encoding_vectors(
      prefix_tokenizer,
      prefix_output,
      {0, 0, 0, 0, 0, 0, 0},
      word_ids({0, 1, 2, 3, 4, 5, 5}));
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_byte_level_basic(data_dir);
  test_byte_level_unicode(data_dir);
  test_replace_regex_before_byte_level_bpe(data_dir);
  test_split_regex_before_byte_level_bpe(data_dir);
  test_byte_level_double_sequence(data_dir);
  test_byte_level_pre_tokenized_sequence(data_dir);
  test_added_token_lstrip_byte_level(data_dir);
  test_added_token_rstrip_byte_level(data_dir);
  return 0;
}
