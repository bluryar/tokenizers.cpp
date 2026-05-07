#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
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

json wordlevel_tokenizer_json(json vocab) {
  return {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", {{"type", "WhitespaceSplit"}}},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "WordLevel"},
           {"unk_token", "[UNK]"},
           {"vocab", std::move(vocab)},
       }},
  };
}

tokenizers_cpp::Tokenizer load_wordlevel(const std::string & name, json vocab) {
  const auto path = write_temp_tokenizer_json(
      name,
      wordlevel_tokenizer_json(std::move(vocab)));
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

void test_wordlevel_unknown_fallback() {
  const auto tokenizer = load_wordlevel(
      "tokenizers_cpp_wordlevel_unknown.json",
      {
          {"[UNK]", 0},
          {"hello", 1},
          {"world", 2},
      });

  const auto output = tokenizer.encode("hello missing world", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 0, 2}));
  assert((output.tokens == std::vector<std::string>{"hello", "[UNK]", "world"}));
  assert((output.offsets == offsets({{0, 5}, {6, 13}, {14, 19}})));
  assert((output.word_ids == word_ids({0U, 1U, 2U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
}

void test_wordlevel_missing_unknown_token_fails_on_unknown_input() {
  const auto tokenizer = load_wordlevel(
      "tokenizers_cpp_wordlevel_missing_unknown.json",
      {
          {"hello", 0},
      });

  const auto known = tokenizer.encode("hello", false);
  assert((known.ids == std::vector<std::uint32_t>{0}));
  assert((known.tokens == std::vector<std::string>{"hello"}));

  try {
    (void)tokenizer.encode("missing", false);
    assert(false && "WordLevel encode should fail when unknown input lacks unk_token");
  } catch (const std::runtime_error & error) {
    assert(std::string(error.what()).find("required token missing") != std::string::npos);
    assert(std::string(error.what()).find("[UNK]") != std::string::npos);
  }
}

}  // namespace

int main() {
  test_wordlevel_unknown_fallback();
  test_wordlevel_missing_unknown_token_fails_on_unknown_input();
  return 0;
}
