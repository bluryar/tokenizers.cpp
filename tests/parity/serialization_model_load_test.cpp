#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

json read_json(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open JSON fixture");
  json value;
  input >> value;
  return value;
}

json read_wordpiece_vocab(const std::filesystem::path & path) {
  std::ifstream input(path);
  assert(input && "failed to open WordPiece vocab fixture");

  json vocab = json::object();
  std::string token;
  std::uint32_t id = 0;
  while (std::getline(input, token)) {
    vocab[token] = id++;
  }
  return vocab;
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
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

json tokenizer_json(json model) {
  return {
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

void expect_model_load_fails(
    const std::string & name,
    json model,
    const std::string & message_part) {
  const auto path = write_temp_tokenizer_json(name, tokenizer_json(std::move(model)));
  try {
    (void)tokenizers_cpp::Tokenizer::from_file(path);
    assert(false && "Tokenizer::from_file should have rejected this model JSON");
  } catch (const std::runtime_error & err) {
    assert(std::string(err.what()).find(message_part) != std::string::npos);
  }
  std::filesystem::remove(path);
}

void test_bpe_load(const std::filesystem::path & data_dir) {
  const auto vocab = read_json(data_dir / "gpt2-vocab.json");
  const auto merges = read_bpe_merges(data_dir / "gpt2-merges.txt");
  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_bpe_serde_load.json",
      tokenizer_json({
          {"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", false},
          {"vocab", vocab},
          {"merges", merges},
      }));

  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  assert(tokenizer.get_vocab_size() == vocab.size());
  assert(tokenizer.token_to_id("hello").value() == vocab.at("hello").get<std::uint32_t>());
  assert(
      tokenizer.token_to_id("<|endoftext|>").value() ==
      vocab.at("<|endoftext|>").get<std::uint32_t>());
  assert(
      tokenizer.id_to_token(vocab.at("hello").get<std::uint32_t>()).value() == "hello");
  std::filesystem::remove(path);

  expect_model_load_fails(
      "tokenizers_cpp_bpe_missing_merges.json",
      {
          {"type", "BPE"},
          {"vocab", {{"a", 0}}},
      },
      "BPE model is missing required merges");
  expect_model_load_fails(
      "tokenizers_cpp_bpe_malformed_merges.json",
      {
          {"type", "BPE"},
          {"vocab", {{"a", 0}}},
          {"merges", json::array({json::array({"a"})})},
      },
      "BPE model merge entries");
  expect_model_load_fails(
      "tokenizers_cpp_bpe_malformed_vocab.json",
      {
          {"type", "BPE"},
          {"vocab", {{"a", -1}}},
          {"merges", json::array()},
      },
      "BPE model vocab id");
}

void test_wordpiece_load(const std::filesystem::path & data_dir) {
  const auto vocab = read_wordpiece_vocab(data_dir / "bert-base-uncased-vocab.txt");
  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_wordpiece_serde_load.json",
      tokenizer_json({
          {"type", "WordPiece"},
          {"unk_token", "[UNK]"},
          {"continuing_subword_prefix", "##"},
          {"max_input_chars_per_word", 100},
          {"vocab", vocab},
      }));

  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  assert(tokenizer.get_vocab_size() == vocab.size());
  assert(tokenizer.token_to_id("[UNK]").value() == vocab.at("[UNK]").get<std::uint32_t>());
  assert(tokenizer.token_to_id("[CLS]").value() == vocab.at("[CLS]").get<std::uint32_t>());
  assert(tokenizer.token_to_id("hello").value() == vocab.at("hello").get<std::uint32_t>());
  assert(
      tokenizer.id_to_token(vocab.at("hello").get<std::uint32_t>()).value() == "hello");
  std::filesystem::remove(path);
}

void test_wordlevel_load(const std::filesystem::path & data_dir) {
  const auto vocab = read_json(data_dir / "gpt2-vocab.json");
  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_wordlevel_serde_load.json",
      tokenizer_json({
          {"type", "WordLevel"},
          {"unk_token", "<unk>"},
          {"vocab", vocab},
      }));

  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  assert(tokenizer.get_vocab_size() == vocab.size());
  assert(tokenizer.token_to_id("hello").value() == vocab.at("hello").get<std::uint32_t>());
  assert(
      tokenizer.token_to_id("<|endoftext|>").value() ==
      vocab.at("<|endoftext|>").get<std::uint32_t>());
  assert(
      tokenizer.id_to_token(vocab.at("hello").get<std::uint32_t>()).value() == "hello");
  std::filesystem::remove(path);
}

void test_deserialize_long_file(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "albert-base-v1-tokenizer.json");
  assert(tokenizer.get_vocab_size() == 30000);
  assert(tokenizer.token_to_id("<pad>").value() == 0);
  assert(tokenizer.token_to_id("<unk>").value() == 1);
  assert(tokenizer.token_to_id("[CLS]").value() == 2);
  assert(tokenizer.id_to_token(3).value() == "[SEP]");
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_bpe_load(data_dir);
  test_wordpiece_load(data_dir);
  test_wordlevel_load(data_dir);
  test_deserialize_long_file(data_dir);
  return 0;
}
