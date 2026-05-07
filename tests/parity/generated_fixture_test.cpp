#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef TOKENIZERS_CPP_TEST_DATA_DIR
#error "TOKENIZERS_CPP_TEST_DATA_DIR must be defined"
#endif

namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open fixture JSON");
  json value;
  input >> value;
  return value;
}

std::vector<std::uint32_t> uint32_vector(const json & value) {
  std::vector<std::uint32_t> result;
  for (const auto & item : value) {
    result.push_back(item.get<std::uint32_t>());
  }
  return result;
}

std::vector<std::string> string_vector(const json & value) {
  std::vector<std::string> result;
  for (const auto & item : value) {
    result.push_back(item.get<std::string>());
  }
  return result;
}

std::vector<tokenizers_cpp::Offset> offset_vector(const json & value) {
  std::vector<tokenizers_cpp::Offset> result;
  for (const auto & item : value) {
    result.push_back(tokenizers_cpp::Offset{
        item.at("start").get<std::size_t>(),
        item.at("end").get<std::size_t>()});
  }
  return result;
}

std::vector<std::optional<std::uint32_t>> word_id_vector(const json & value) {
  std::vector<std::optional<std::uint32_t>> result;
  for (const auto & item : value) {
    if (item.is_null()) {
      result.push_back(std::nullopt);
    } else {
      result.push_back(item.get<std::uint32_t>());
    }
  }
  return result;
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_TEST_DATA_DIR);
  const auto fixture_path = data_dir / "simple_wordlevel_fixture.json";
  const auto fixture = read_json(fixture_path);
  const auto tokenizer_path = data_dir / fixture.at("tokenizer_json").get<std::string>();
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(tokenizer_path);

  for (const auto & test_case : fixture.at("cases")) {
    const auto encoded = tokenizer.encode(
        test_case.at("text").get<std::string>(),
        test_case.value("add_special_tokens", true));
    const auto & expected = test_case.at("encoding");

    assert(encoded.ids == uint32_vector(expected.at("ids")));
    assert(encoded.tokens == string_vector(expected.at("tokens")));
    assert(encoded.offsets == offset_vector(expected.at("offsets")));
    assert(encoded.type_ids == uint32_vector(expected.at("type_ids")));
    assert(encoded.word_ids == word_id_vector(expected.at("word_ids")));
    assert(encoded.special_tokens_mask == uint32_vector(expected.at("special_tokens_mask")));
    assert(encoded.attention_mask == uint32_vector(expected.at("attention_mask")));
    assert(tokenizer.decode(encoded.ids, true) == test_case.at("decode").get<std::string>());
  }

  return 0;
}
