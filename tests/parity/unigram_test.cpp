#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef TOKENIZERS_CPP_HF_TEST_DATA_DIR
#error "TOKENIZERS_CPP_HF_TEST_DATA_DIR must be defined"
#endif

namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open JSON fixture");
  json value;
  input >> value;
  return value;
}

json tokenizer_json(json model) {
  return json{
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", nullptr},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model", std::move(model)},
  };
}

std::filesystem::path write_temp_tokenizer_json(
    const std::string & name,
    const json & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

tokenizers_cpp::Tokenizer load_unigram(const std::string & name, json model) {
  const auto path = write_temp_tokenizer_json(name, tokenizer_json(std::move(model)));
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

std::vector<tokenizers_cpp::Offset> contiguous_offsets(
    const std::vector<std::string> & tokens) {
  std::vector<tokenizers_cpp::Offset> offsets;
  offsets.reserve(tokens.size());
  std::size_t pos = 0;
  for (const auto & token : tokens) {
    offsets.push_back(tokenizers_cpp::Offset{pos, pos + token.size()});
    pos += token.size();
  }
  return offsets;
}

std::vector<std::optional<std::uint32_t>> repeated_word_ids(
    std::size_t count,
    std::uint32_t word_id) {
  return std::vector<std::optional<std::uint32_t>>(count, word_id);
}

void check_common_vectors(const tokenizers_cpp::Encoding & output) {
  assert(output.type_ids == std::vector<std::uint32_t>(output.ids.size(), 0));
  assert(output.special_tokens_mask == std::vector<std::uint32_t>(output.ids.size(), 0));
  assert(output.attention_mask == std::vector<std::uint32_t>(output.ids.size(), 1));
  assert(output.word_ids == repeated_word_ids(output.ids.size(), 0));
}

void test_unigram_from_file_fixture(const std::filesystem::path & data_dir) {
  auto model = read_json(data_dir / "unigram.json");
  model["type"] = "Unigram";
  const auto tokenizer = load_unigram(
      "tokenizers_cpp_unigram_from_file_fixture.json",
      std::move(model));

  const std::string input =
      "\xE5\x90\xBE\xE8\xBC\xA9\xE3\x80\x8A\xE3\x82\x8F\xE3\x81\x8C"
      "\xE3\x81\xAF\xE3\x81\x84\xE3\x80\x8B\xE3\x81\xAF\xE7\x8C\xAB"
      "\xE3\x81\xA7\xE3\x81\x82\xE3\x82\x8B\xE3\x80\x82\xE5\x90\x8D"
      "\xE5\x89\x8D\xE3\x81\xAF\xE3\x81\xBE\xE3\x81\xA0\xE7\x84\xA1"
      "\xE3\x81\x84\xE3\x80\x82";
  const std::vector<std::string> expected_tokens = {
      "\xE5\x90\xBE\xE8\xBC\xA9",
      "\xE3\x80\x8A",
      "\xE3\x82\x8F\xE3\x81\x8C",
      "\xE3\x81\xAF\xE3\x81\x84",
      "\xE3\x80\x8B",
      "\xE3\x81\xAF",
      "\xE7\x8C\xAB",
      "\xE3\x81\xA7\xE3\x81\x82\xE3\x82\x8B",
      "\xE3\x80\x82",
      "\xE5\x90\x8D\xE5\x89\x8D",
      "\xE3\x81\xAF\xE3\x81\xBE\xE3\x81\xA0",
      "\xE7\x84\xA1\xE3\x81\x84",
      "\xE3\x80\x82",
  };

  const auto output = tokenizer.encode(input, false);
  assert(output.tokens == expected_tokens);
  assert(output.offsets == contiguous_offsets(expected_tokens));
  check_common_vectors(output);
}

void test_best_path_and_fused_unknown() {
  const auto tokenizer = load_unigram(
      "tokenizers_cpp_unigram_best_path.json",
      {
          {"type", "Unigram"},
          {"unk_id", 0},
          {"byte_fallback", false},
          {"vocab",
           json::array({
               json::array({"<unk>", 0.0}),
               json::array({"a", 0.0}),
               json::array({"b", 0.0}),
               json::array({"c", 0.0}),
               json::array({"d", 0.0}),
               json::array({"cd", 1.0}),
               json::array({"ab", 2.0}),
               json::array({"abc", 5.0}),
               json::array({"abcd", 10.0}),
           })},
      });

  const auto output = tokenizer.encode("abcdacdxx", false);
  assert((output.ids == std::vector<std::uint32_t>{8, 1, 5, 0}));
  assert((output.tokens == std::vector<std::string>{"abcd", "a", "cd", "xx"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 4}, {4, 5}, {5, 7}, {7, 9}}));
  check_common_vectors(output);
}

void test_byte_fallback() {
  const auto tokenizer = load_unigram(
      "tokenizers_cpp_unigram_byte_fallback.json",
      {
          {"type", "Unigram"},
          {"unk_id", 0},
          {"byte_fallback", true},
          {"vocab",
           json::array({
               json::array({"<unk>", 0.0}),
               json::array({"<0xC3>", -0.01}),
               json::array({"<0xA9>", -0.03}),
           })},
      });

  const auto output = tokenizer.encode("\xC3\xA9", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"<0xC3>", "<0xA9>"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{{0, 2}, {0, 2}}));
  check_common_vectors(output);

  const auto unknown = tokenizer.encode("?", false);
  assert((unknown.ids == std::vector<std::uint32_t>{0}));
  assert((unknown.tokens == std::vector<std::string>{"?"}));
  assert((unknown.offsets == std::vector<tokenizers_cpp::Offset>{{0, 1}}));
  check_common_vectors(unknown);
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_unigram_from_file_fixture(data_dir);
  test_best_path_and_fused_unknown();
  test_byte_fallback();
  return 0;
}
