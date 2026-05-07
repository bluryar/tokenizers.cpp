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

json tokenizer_json(json model) {
  return json{
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", false},
           {"trim_offsets", false},
           {"use_regex", false},
       }},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model", std::move(model)},
  };
}

json tokenizer_json(json model, json pre_tokenizer) {
  auto value = tokenizer_json(std::move(model));
  value["pre_tokenizer"] = std::move(pre_tokenizer);
  return value;
}

std::filesystem::path write_temp_tokenizer_json(
    const std::string & name,
    const json & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

std::filesystem::path write_temp_text_file(
    const std::string & name,
    const std::string & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

tokenizers_cpp::Tokenizer load_tokenizer(const std::string & name, json model) {
  const auto path = write_temp_tokenizer_json(name, tokenizer_json(std::move(model)));
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_tokenizer(
    const std::string & name,
    json model,
    json pre_tokenizer) {
  const auto path = write_temp_tokenizer_json(
      name,
      tokenizer_json(std::move(model), std::move(pre_tokenizer)));
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_bpe_files(
    const std::string & name,
    const std::string & vocab,
    const std::string & merges,
    const tokenizers_cpp::BpeOptions & options = {}) {
  const auto vocab_path = write_temp_text_file(name + "_vocab.json", vocab);
  const auto merges_path = write_temp_text_file(name + "_merges.txt", merges);
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_bpe_files(vocab_path, merges_path, options);
  std::filesystem::remove(vocab_path);
  std::filesystem::remove(merges_path);
  return tokenizer;
}

void assert_offsets(
    const tokenizers_cpp::Encoding & encoding,
    const std::vector<tokenizers_cpp::Offset> & expected) {
  assert(encoding.offsets == expected);
}

void test_unk_not_fused() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_unk_not_fused.json",
      {
          {"type", "BPE"},
          {"unk_token", "<unk>"},
          {"fuse_unk", false},
          {"vocab", {{"<unk>", 0}, {"a", 1}, {"b", 2}}},
          {"merges", json::array()},
      });

  const auto output = tokenizer.encode("accb", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 0, 0, 2}));
  assert((output.tokens == std::vector<std::string>{"a", "<unk>", "<unk>", "b"}));
  assert_offsets(output, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  assert((output.word_ids == std::vector<std::optional<std::uint32_t>>{0, 0, 0, 0}));
}

void test_unk_fused() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_unk_fused.json",
      {
          {"type", "BPE"},
          {"unk_token", "<unk>"},
          {"fuse_unk", true},
          {"vocab", {{"<unk>", 0}, {"a", 1}, {"b", 2}}},
          {"merges", json::array()},
      });

  const auto output = tokenizer.encode("accb", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 0, 2}));
  assert((output.tokens == std::vector<std::string>{"a", "<unk>", "b"}));
  assert_offsets(output, {{0, 1}, {1, 3}, {3, 4}});
  assert((output.word_ids == std::vector<std::optional<std::uint32_t>>{0, 0, 0}));
}

void test_missing_unk_fails_on_unknown() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_missing_unk.json",
      {
          {"type", "BPE"},
          {"unk_token", "<unk>"},
          {"vocab", {{"a", 0}}},
          {"merges", json::array()},
      });

  bool threw = false;
  try {
    (void)tokenizer.encode("z", false);
  } catch (const std::runtime_error & error) {
    threw = std::string(error.what()).find("<unk>") != std::string::npos;
  }
  assert(threw);
}

void test_ignore_merges_prefers_full_vocab_hit() {
  const json base_model = {
      {"type", "BPE"},
      {"vocab", {{"a", 0}, {"b", 1}, {"ab", 2}}},
      {"merges", json::array()},
  };

  auto model_with_ignore = base_model;
  model_with_ignore["ignore_merges"] = true;
  const auto ignore = load_tokenizer(
      "tokenizers_cpp_bpe_ignore_merges_true.json",
      std::move(model_with_ignore));
  const auto ignored = ignore.encode("ab", false);
  assert((ignored.ids == std::vector<std::uint32_t>{2}));
  assert((ignored.tokens == std::vector<std::string>{"ab"}));
  assert_offsets(ignored, {{0, 2}});

  auto model_without_ignore = base_model;
  model_without_ignore["ignore_merges"] = false;
  const auto regular = load_tokenizer(
      "tokenizers_cpp_bpe_ignore_merges_false.json",
      std::move(model_without_ignore));
  const auto merged = regular.encode("ab", false);
  assert((merged.ids == std::vector<std::uint32_t>{0, 1}));
  assert((merged.tokens == std::vector<std::string>{"a", "b"}));
  assert_offsets(merged, {{0, 1}, {1, 2}});
}

void test_continuing_prefix_merges_subword() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_prefix_merge.json",
      {
          {"type", "BPE"},
          {"continuing_subword_prefix", "##"},
          {"vocab", {{"a", 0}, {"##b", 1}, {"ab", 2}}},
          {"merges", json::array({json::array({"a", "##b"})})},
      });

  const auto output = tokenizer.encode("ab", false);
  assert((output.ids == std::vector<std::uint32_t>{2}));
  assert((output.tokens == std::vector<std::string>{"ab"}));
  assert_offsets(output, {{0, 2}});
}

void test_end_of_word_suffix() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_end_suffix.json",
      {
          {"type", "BPE"},
          {"end_of_word_suffix", "</w>"},
          {"vocab", {{"a", 0}, {"b</w>", 1}}},
          {"merges", json::array()},
      });

  const auto output = tokenizer.encode("ab", false);
  assert((output.ids == std::vector<std::uint32_t>{0, 1}));
  assert((output.tokens == std::vector<std::string>{"a", "b</w>"}));
  assert_offsets(output, {{0, 1}, {1, 2}});
}

void test_byte_fallback() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_byte_fallback.json",
      {
          {"type", "BPE"},
          {"unk_token", "<unk>"},
          {"byte_fallback", true},
          {"vocab", {{"<unk>", 0}, {"<0x61>", 1}, {"<0x62>", 2}}},
          {"merges", json::array()},
      });

  const auto output = tokenizer.encode("ab", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"<0x61>", "<0x62>"}));
  assert_offsets(output, {{0, 1}, {1, 2}});

  const auto unknown = tokenizer.encode("c", false);
  assert((unknown.ids == std::vector<std::uint32_t>{0}));
  assert((unknown.tokens == std::vector<std::string>{"<unk>"}));
  assert_offsets(unknown, {{0, 1}});
}

void test_raw_bpe_byte_fallback_newline() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_raw_bpe_byte_fallback_newline.json",
      {
          {"type", "BPE"},
          {"unk_token", "<unk>"},
          {"byte_fallback", true},
          {"vocab", {{"<unk>", 0}, {"<0x0A>", 1}}},
          {"merges", json::array()},
      },
      nullptr);

  const auto output = tokenizer.encode("\n", false);
  assert((output.ids == std::vector<std::uint32_t>{1}));
  assert((output.tokens == std::vector<std::string>{"<0x0A>"}));
  assert_offsets(output, {{0, 1}});
  assert((output.word_ids == std::vector<std::optional<std::uint32_t>>{0}));
}

void test_raw_bpe_cache_does_not_cross_instances() {
  const auto tokenizer_a = load_tokenizer(
      "tokenizers_cpp_raw_bpe_cache_instance_a.json",
      {
          {"type", "BPE"},
          {"vocab",
           {
               {"h", 0},
               {"e", 1},
               {"l", 2},
               {"o", 3},
               {"he", 4},
               {"hel", 5},
               {"hell", 6},
               {"hello", 7},
           }},
          {"merges",
           json::array({
               json::array({"h", "e"}),
               json::array({"he", "l"}),
               json::array({"hel", "l"}),
               json::array({"hell", "o"}),
           })},
      },
      nullptr);
  const auto tokenizer_b = load_tokenizer(
      "tokenizers_cpp_raw_bpe_cache_instance_b.json",
      {
          {"type", "BPE"},
          {"vocab", {{"h", 0}, {"e", 1}, {"l", 2}, {"o", 3}}},
          {"merges", json::array()},
      },
      nullptr);

  const auto a = tokenizer_a.encode("hello", false);
  const auto b = tokenizer_b.encode("hello", false);
  const auto a2 = tokenizer_a.encode("hello", false);
  const auto b2 = tokenizer_b.encode("hello", false);

  assert((a.ids == std::vector<std::uint32_t>{7}));
  assert((a.tokens == std::vector<std::string>{"hello"}));
  assert_offsets(a, {{0, 5}});
  assert((b.ids == std::vector<std::uint32_t>{0, 1, 2, 2, 3}));
  assert((b.tokens == std::vector<std::string>{"h", "e", "l", "l", "o"}));
  assert_offsets(b, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}});
  assert(a2.ids == a.ids);
  assert(a2.tokens == a.tokens);
  assert(b2.ids == b.ids);
  assert(b2.tokens == b.tokens);
}

void test_merge_token_oov_rejected_on_load() {
  bool threw = false;
  try {
    (void)load_tokenizer(
        "tokenizers_cpp_bpe_merge_token_oov.json",
        {
            {"type", "BPE"},
            {"vocab", {{"a", 0}, {"b", 1}, {"ab", 2}}},
            {"merges",
             json::array({
                 json::array({"a", "b"}),
                 json::array({"a", "d"}),
             })},
        });
  } catch (const std::runtime_error & error) {
    threw = std::string(error.what()).find("out of vocabulary") != std::string::npos;
  }
  assert(threw);
}

void test_bpe_from_files_public_api() {
  const auto tokenizer = load_bpe_files(
      "tokenizers_cpp_bpe_from_files",
      R"json({"a":0,"b":1,"c":2,"ab":3,"abc":4})json",
      "#version: 0.2\n"
      "a b\n"
      "ab c\n");

  const auto output = tokenizer.encode("abc", false);
  assert((output.ids == std::vector<std::uint32_t>{4}));
  assert((output.tokens == std::vector<std::string>{"abc"}));
  assert_offsets(output, {{0, 3}});
}

void test_bpe_from_files_options() {
  tokenizers_cpp::BpeOptions options;
  options.dropout = 1.0;
  const auto tokenizer = load_bpe_files(
      "tokenizers_cpp_bpe_from_files_options",
      R"json({"a":0,"b":1,"c":2,"ab":3,"abc":4})json",
      "#version: 0.2\n"
      "a b\n"
      "ab c\n",
      options);

  const auto output = tokenizer.encode("abc", false);
  assert((output.ids == std::vector<std::uint32_t>{0, 1, 2}));
  assert((output.tokens == std::vector<std::string>{"a", "b", "c"}));
  assert_offsets(output, {{0, 1}, {1, 2}, {2, 3}});
}

void test_bpe_from_files_bad_merges_rejected() {
  bool threw = false;
  try {
    (void)load_bpe_files(
        "tokenizers_cpp_bpe_from_files_bad_merges",
        R"json({"a":0,"b":1,"c":2,"ab":3})json",
        "#version: 0.2\n"
        "a b\n"
        "c\n");
  } catch (const std::runtime_error & error) {
    threw = std::string(error.what()).find("invalid at line 2") != std::string::npos;
  }
  assert(threw);
}

void test_bpe_from_files_merge_token_oov_rejected() {
  bool threw = false;
  try {
    (void)load_bpe_files(
        "tokenizers_cpp_bpe_from_files_merge_oov",
        R"json({"a":0,"b":1,"ab":2})json",
        "#version: 0.2\n"
        "a b\n"
        "a d\n");
  } catch (const std::runtime_error & error) {
    threw = std::string(error.what()).find("out of vocabulary") != std::string::npos;
  }
  assert(threw);
}

json dropout_fixture_model(json dropout) {
  return {
      {"type", "BPE"},
      {"dropout", std::move(dropout)},
      {"vocab",
       {
           {"u", 0},
           {"n", 1},
           {"r", 2},
           {"e", 3},
           {"l", 4},
           {"a", 5},
           {"t", 6},
           {"d", 7},
           {"re", 8},
           {"at", 9},
           {"ed", 10},
           {"un", 11},
           {"ated", 12},
           {"rel", 13},
           {"related", 14},
           {"unrelated", 15},
       }},
      {"merges",
       json::array({
           json::array({"r", "e"}),
           json::array({"a", "t"}),
           json::array({"e", "d"}),
           json::array({"u", "n"}),
           json::array({"at", "ed"}),
           json::array({"re", "l"}),
           json::array({"rel", "ated"}),
           json::array({"un", "related"}),
       })},
  };
}

void test_dropout_null_and_zero_keep_deterministic_merges() {
  const auto null_dropout = load_tokenizer(
      "tokenizers_cpp_bpe_dropout_null.json",
      dropout_fixture_model(nullptr));
  const auto null_output = null_dropout.encode("unrelated", false);
  assert((null_output.ids == std::vector<std::uint32_t>{15}));
  assert((null_output.tokens == std::vector<std::string>{"unrelated"}));
  assert_offsets(null_output, {{0, 9}});

  const auto zero_dropout = load_tokenizer(
      "tokenizers_cpp_bpe_dropout_zero.json",
      dropout_fixture_model(0.0));
  const auto zero_output = zero_dropout.encode("unrelated", false);
  assert((zero_output.ids == std::vector<std::uint32_t>{15}));
  assert((zero_output.tokens == std::vector<std::string>{"unrelated"}));
  assert_offsets(zero_output, {{0, 9}});
}

void test_dropout_one_skips_all_merges() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_dropout_one.json",
      dropout_fixture_model(1.0));

  const auto output = tokenizer.encode("unrelated", false);
  assert((output.ids == std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5, 6, 3, 7}));
  assert((output.tokens == std::vector<std::string>{"u", "n", "r", "e", "l", "a", "t", "e", "d"}));
  assert_offsets(output, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}});
}

void test_dropout_middle_probability_shape() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_bpe_dropout_middle.json",
      dropout_fixture_model(0.5));

  for (std::size_t iteration = 0; iteration < 16; ++iteration) {
    const auto output = tokenizer.encode("unrelated", false);
    assert(!output.ids.empty());
    assert(output.ids.size() <= 9);
    assert(output.tokens.size() == output.ids.size());
    assert(output.offsets.size() == output.ids.size());
    assert(output.offsets.front().start == 0);
    assert(output.offsets.back().end == 9);
    for (const auto id : output.ids) {
      assert(id <= 15);
    }
  }
}

void test_invalid_dropout_rejected() {
  bool threw = false;
  try {
    (void)load_tokenizer(
        "tokenizers_cpp_bpe_dropout_invalid.json",
        dropout_fixture_model(1.5));
  } catch (const std::runtime_error & error) {
    threw = std::string(error.what()).find("dropout") != std::string::npos;
  }
  assert(threw);
}

}  // namespace

int main() {
  test_unk_not_fused();
  test_unk_fused();
  test_missing_unk_fails_on_unknown();
  test_ignore_merges_prefers_full_vocab_hit();
  test_continuing_prefix_merges_subword();
  test_end_of_word_suffix();
  test_byte_fallback();
  test_raw_bpe_byte_fallback_newline();
  test_raw_bpe_cache_does_not_cross_instances();
  test_merge_token_oov_rejected_on_load();
  test_bpe_from_files_public_api();
  test_bpe_from_files_options();
  test_bpe_from_files_bad_merges_rejected();
  test_bpe_from_files_merge_token_oov_rejected();
  test_dropout_null_and_zero_keep_deterministic_merges();
  test_dropout_one_skips_all_merges();
  test_dropout_middle_probability_shape();
  test_invalid_dropout_rejected();
  return 0;
}
