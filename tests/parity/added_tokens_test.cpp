#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_tokenizer_json(const std::string & name, const std::string & json) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path);
  out << json;
  return path;
}

std::string empty_bpe_tokenizer_json(const std::string & added_tokens_json) {
  return R"json({
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": )json" + added_tokens_json + R"json(,
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "BPE",
      "vocab": {},
      "merges": []
    }
  })json";
}

std::string wordlevel_tokenizer_json(
    const std::string & vocab_json,
    const std::string & added_tokens_json) {
  return R"json({
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": )json" + added_tokens_json + R"json(,
    "normalizer": null,
    "pre_tokenizer": {
      "type": "WhitespaceSplit"
    },
    "post_processor": null,
    "decoder": null,
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": )json" + vocab_json + R"json(
    }
  })json";
}

tokenizers_cpp::Tokenizer load_temp_tokenizer(
    const std::string & name,
    const std::string & json) {
  const auto path = write_tokenizer_json(name, json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void expect_load_fails(
    const std::string & name,
    const std::string & json,
    const std::string & message_part) {
  const auto path = write_tokenizer_json(name, json);
  try {
    (void)tokenizers_cpp::Tokenizer::from_file(path);
    assert(false && "Tokenizer::from_file should have rejected this added_tokens JSON");
  } catch (const std::runtime_error & err) {
    assert(std::string(err.what()).find(message_part) != std::string::npos);
  }
  std::filesystem::remove(path);
}

std::vector<tokenizers_cpp::Offset> offsets(std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

void test_json_loaded_added_tokens_assign_runtime_ids() {
  const auto tokenizer = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_ids.json",
      empty_bpe_tokenizer_json(R"json([
        {"id": 0, "content": "<cls>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 1, "content": "<sep>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "hello", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
        {"id": 3, "content": "world", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false}
      ])json"));

  assert(tokenizer.get_vocab_size() == 4);
  assert(tokenizer.token_to_id("<cls>").value() == 0);
  assert(tokenizer.token_to_id("<sep>").value() == 1);
  assert(tokenizer.token_to_id("hello").value() == 2);
  assert(tokenizer.token_to_id("world").value() == 3);
  assert(tokenizer.id_to_token(0).value() == "<cls>");
  assert(tokenizer.id_to_token(3).value() == "world");

  const auto encoded = tokenizer.encode("<cls> hello world <sep>", false);
  assert((encoded.ids == std::vector<std::uint32_t>{0, 2, 3, 1}));
  assert((encoded.tokens == std::vector<std::string>{"<cls>", "hello", "world", "<sep>"}));
  assert((encoded.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(tokenizer.decode(encoded.ids, true) == "hello world");
  assert(tokenizer.decode(encoded.ids, false) == "<cls> hello world <sep>");
}

void test_json_loaded_added_tokens_do_not_trust_mismatched_ids() {
  const auto tokenizer = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_mismatched_ids.json",
      wordlevel_tokenizer_json(
          R"json({"hello": 0})json",
          R"json([
            {"id": 42, "content": "new", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": false},
            {"id": 99, "content": "hello", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}
          ])json"));

  assert(tokenizer.get_vocab_size() == 2);
  assert(tokenizer.token_to_id("hello").value() == 0);
  assert(tokenizer.token_to_id("new").value() == 1);
  assert(tokenizer.id_to_token(1).value() == "new");
  assert((tokenizer.encode("hello new", false).special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
}

void test_lstrip_and_rstrip_matching() {
  const std::string vocab = R"json({"[UNK]": 0, "hello": 1, "world": 2})json";
  const auto lstrip_tokenizer = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_lstrip.json",
      wordlevel_tokenizer_json(vocab, R"json([
        {"id": 3, "content": "<mask>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true}
      ])json"));

  const auto lstrip = lstrip_tokenizer.encode("hello <mask> world", false);
  assert((lstrip.ids == std::vector<std::uint32_t>{1, 3, 2}));
  assert((lstrip.tokens == std::vector<std::string>{"hello", " <mask>", "world"}));
  assert((lstrip.offsets == offsets({{0, 5}, {5, 12}, {13, 18}})));
  assert((lstrip.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));

  const auto rstrip_tokenizer = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_rstrip.json",
      wordlevel_tokenizer_json(vocab, R"json([
        {"id": 3, "content": "<mask>", "single_word": false, "lstrip": false, "rstrip": true, "normalized": false, "special": true}
      ])json"));

  const auto rstrip = rstrip_tokenizer.encode("hello <mask> world", false);
  assert((rstrip.ids == std::vector<std::uint32_t>{1, 3, 2}));
  assert((rstrip.tokens == std::vector<std::string>{"hello", "<mask> ", "world"}));
  assert((rstrip.offsets == offsets({{0, 5}, {6, 13}, {13, 18}})));
  assert((rstrip.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
}

void test_single_word_matching() {
  const std::string vocab =
      R"json({"[UNK]": 0, "I": 1, "like": 2, "dancing": 3, "danc": 4})json";
  const auto single_word = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_single_word.json",
      wordlevel_tokenizer_json(vocab, R"json([
        {"id": 5, "content": "ing", "single_word": true, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ])json"));

  const auto not_inside_word = single_word.encode("I like dancing ing", false);
  assert((not_inside_word.ids == std::vector<std::uint32_t>{1, 2, 3, 5}));
  assert((not_inside_word.tokens == std::vector<std::string>{"I", "like", "dancing", "ing"}));
  assert((not_inside_word.offsets == offsets({{0, 1}, {2, 6}, {7, 14}, {15, 18}})));

  const auto any_position = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_inside_word.json",
      wordlevel_tokenizer_json(vocab, R"json([
        {"id": 5, "content": "ing", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ])json"));

  const auto inside_word = any_position.encode("I like dancing", false);
  assert((inside_word.ids == std::vector<std::uint32_t>{1, 2, 4, 5}));
  assert((inside_word.tokens == std::vector<std::string>{"I", "like", "danc", "ing"}));
  assert((inside_word.offsets == offsets({{0, 1}, {2, 6}, {7, 11}, {11, 14}})));
}

void test_leftmost_longest_overlap_matching() {
  const std::string vocab = R"json({"[UNK]": 0, "I": 1, "like": 2, "l": 3})json";
  const auto tokenizer = load_temp_tokenizer(
      "tokenizers_cpp_added_tokens_overlap.json",
      wordlevel_tokenizer_json(vocab, R"json([
        {"id": 4, "content": "nci", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 5, "content": "danc", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 6, "content": "ing", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 7, "content": "ike", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ])json"));

  const auto encoded = tokenizer.encode("I like dancing", false);
  assert((encoded.ids == std::vector<std::uint32_t>{1, 3, 7, 5, 6}));
  assert((encoded.tokens == std::vector<std::string>{"I", "l", "ike", "danc", "ing"}));
  assert((encoded.offsets == offsets({{0, 1}, {2, 3}, {3, 6}, {7, 11}, {11, 14}})));
}

void test_added_tokens_validation() {
  expect_load_fails(
      "tokenizers_cpp_added_tokens_not_array.json",
      empty_bpe_tokenizer_json(R"json({"id": 0, "content": "<cls>"})json"),
      "added_tokens must be an array");
  expect_load_fails(
      "tokenizers_cpp_added_tokens_missing_content.json",
      empty_bpe_tokenizer_json(R"json([{"id": 0, "special": true}])json"),
      "string content");
  expect_load_fails(
      "tokenizers_cpp_added_tokens_bad_flag.json",
      empty_bpe_tokenizer_json(R"json([
        {"id": 0, "content": "<cls>", "single_word": "no"}
      ])json"),
      "single_word must be a boolean");
}

}  // namespace

int main() {
  test_json_loaded_added_tokens_assign_runtime_ids();
  test_json_loaded_added_tokens_do_not_trust_mismatched_ids();
  test_lstrip_and_rstrip_matching();
  test_single_word_matching();
  test_leftmost_longest_overlap_matching();
  test_added_tokens_validation();
  return 0;
}
