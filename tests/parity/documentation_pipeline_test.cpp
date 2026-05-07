#include "tokenizers_cpp/tokenizer.hpp"

#include <algorithm>
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

std::filesystem::path write_temp_tokenizer_json(
    const std::string & name,
    const json & value) {
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

void install_quicktour_template_processing(json & tokenizer_json) {
  tokenizer_json["post_processor"] = {
      {"type", "TemplateProcessing"},
      {"single",
       json::array({
           {{"SpecialToken", {{"id", "[CLS]"}, {"type_id", 0}}}},
           {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
           {{"SpecialToken", {{"id", "[SEP]"}, {"type_id", 0}}}},
       })},
      {"pair",
       json::array({
           {{"SpecialToken", {{"id", "[CLS]"}, {"type_id", 0}}}},
           {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
           {{"SpecialToken", {{"id", "[SEP]"}, {"type_id", 0}}}},
           {{"Sequence", {{"id", "B"}, {"type_id", 1}}}},
           {{"SpecialToken", {{"id", "[SEP]"}, {"type_id", 1}}}},
       })},
      {"special_tokens",
       {
           {"[CLS]",
            {
                {"id", "[CLS]"},
                {"ids", json::array({1})},
                {"tokens", json::array({"[CLS]"})},
            }},
           {"[SEP]",
            {
                {"id", "[SEP]"},
                {"ids", json::array({2})},
                {"tokens", json::array({"[SEP]"})},
           }},
       }},
  };
}

void install_quicktour_padding(json & tokenizer_json) {
  tokenizer_json["padding"] = {
      {"strategy", "BatchLongest"},
      {"direction", "Right"},
      {"pad_to_multiple_of", nullptr},
      {"pad_id", 3},
      {"pad_type_id", 0},
      {"pad_token", "[PAD]"},
  };
}

tokenizers_cpp::Tokenizer load_tokenizer_from_json(
    const std::string & name,
    const json & tokenizer_json) {
  const auto path = write_temp_tokenizer_json(name, tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_documentation_pipeline_tokenizer(
    const std::filesystem::path & data_dir) {
  auto tokenizer_json = read_json(data_dir / "tokenizer-wiki.json");
  tokenizer_json["normalizer"] = {
      {"type", "Sequence"},
      {"normalizers",
       json::array({
           {{"type", "NFD"}},
           {{"type", "StripAccents"}},
       })},
  };
  tokenizer_json["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array({
           {{"type", "Whitespace"}},
           {{"type", "Digits"}, {"individual_digits", true}},
       })},
  };
  install_quicktour_template_processing(tokenizer_json);

  return load_tokenizer_from_json(
      "tokenizers_cpp_documentation_pipeline.json",
      tokenizer_json);
}

tokenizers_cpp::Tokenizer load_quicktour_tokenizer(
    const std::filesystem::path & data_dir,
    bool with_template,
    bool with_padding) {
  auto tokenizer_json = read_json(data_dir / "tokenizer-wiki.json");
  if (with_template) {
    install_quicktour_template_processing(tokenizer_json);
  }
  if (with_padding) {
    install_quicktour_padding(tokenizer_json);
  }
  return load_tokenizer_from_json(
      with_padding
          ? "tokenizers_cpp_documentation_quicktour_padding.json"
          : "tokenizers_cpp_documentation_quicktour.json",
      tokenizer_json);
}

void test_documentation_pipeline(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_documentation_pipeline_tokenizer(data_dir);
  const auto output = tokenizer.encode(
      "Hello, y'all! How are you \xF0\x9F\x98\x81 ?",
      true);

  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27253,
                            16,
                            93,
                            11,
                            5097,
                            5,
                            7961,
                            5112,
                            6218,
                            0,
                            35,
                            2,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "Hello",
                               ",",
                               "y",
                               "'",
                               "all",
                               "!",
                               "How",
                               "are",
                               "you",
                               "[UNK]",
                               "?",
                               "[SEP]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {5, 6},
                              {7, 8},
                              {8, 9},
                              {9, 12},
                              {12, 13},
                              {14, 17},
                              {18, 21},
                              {22, 25},
                              {26, 30},
                              {31, 32},
                              {0, 0},
                          })));
  assert((output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                std::nullopt,
                            })));
  assert((output.type_ids == std::vector<std::uint32_t>{
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            }));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{
                                            1,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            1,
                                        }));
  assert((output.attention_mask == std::vector<std::uint32_t>{
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                  }));
  assert(tokenizer.decode(output.ids, true) ==
      "Hello , y ' all ! How are you ?");
}

void test_quicktour_initial_encode(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_quicktour_tokenizer(data_dir, false, false);
  const auto output = tokenizer.encode(
      "Hello, y'all! How are you \xF0\x9F\x98\x81 ?",
      true);

  assert((output.ids == std::vector<std::uint32_t>{
                            27253,
                            16,
                            93,
                            11,
                            5097,
                            5,
                            7961,
                            5112,
                            6218,
                            0,
                            35,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "Hello",
                               ",",
                               "y",
                               "'",
                               "all",
                               "!",
                               "How",
                               "are",
                               "you",
                               "[UNK]",
                               "?",
                           }));
  assert((output.offsets == offsets({
                              {0, 5},
                              {5, 6},
                              {7, 8},
                              {8, 9},
                              {9, 12},
                              {12, 13},
                              {14, 17},
                              {18, 21},
                              {22, 25},
                              {26, 30},
                              {31, 32},
                          })));
  assert(output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U}));
  assert((output.type_ids == std::vector<std::uint32_t>{
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                            }));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                        }));
  assert((output.attention_mask == std::vector<std::uint32_t>{
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                  }));
  assert(output.offsets.at(9) == (tokenizers_cpp::Offset{26, 30}));
  assert(tokenizer.token_to_id("[SEP]").value() == 2);
}

void test_quicktour_template_single_pair_and_batch(
    const std::filesystem::path & data_dir) {
  const auto tokenizer = load_quicktour_tokenizer(data_dir, true, false);

  const auto single = tokenizer.encode(
      "Hello, y'all! How are you \xF0\x9F\x98\x81 ?",
      true);
  assert((single.tokens == std::vector<std::string>{
                              "[CLS]",
                              "Hello",
                              ",",
                              "y",
                              "'",
                              "all",
                              "!",
                              "How",
                              "are",
                              "you",
                              "[UNK]",
                              "?",
                              "[SEP]",
                          }));
  assert((single.ids == std::vector<std::uint32_t>{
                           1,
                           27253,
                           16,
                           93,
                           11,
                           5097,
                           5,
                           7961,
                           5112,
                           6218,
                           0,
                           35,
                           2,
                       }));

  const auto pair = tokenizer.encode_pair(
      "Hello, y'all!",
      "How are you \xF0\x9F\x98\x81 ?",
      true);
  assert((pair.tokens == std::vector<std::string>{
                            "[CLS]",
                            "Hello",
                            ",",
                            "y",
                            "'",
                            "all",
                            "!",
                            "[SEP]",
                            "How",
                            "are",
                            "you",
                            "[UNK]",
                            "?",
                            "[SEP]",
                        }));
  assert((pair.type_ids == std::vector<std::uint32_t>{
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               1,
                               1,
                               1,
                               1,
                               1,
                               1,
                           }));
  assert((pair.ids == std::vector<std::uint32_t>{
                          1,
                          27253,
                          16,
                          93,
                          11,
                          5097,
                          5,
                          2,
                          7961,
                          5112,
                          6218,
                          0,
                          35,
                          2,
                      }));
  assert((pair.offsets == offsets({
                             {0, 0},
                             {0, 5},
                             {5, 6},
                             {7, 8},
                             {8, 9},
                             {9, 12},
                             {12, 13},
                             {0, 0},
                             {0, 3},
                             {4, 7},
                             {8, 11},
                             {12, 16},
                             {17, 18},
                             {0, 0},
                         })));

  const auto batch = tokenizer.encode_batch(
      std::vector<std::string>{
          "Hello, y'all!",
          "How are you \xF0\x9F\x98\x81 ?",
      },
      true);
  assert(batch.size() == 2);
  assert((batch[0].tokens == std::vector<std::string>{
                               "[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]"}));
  assert((batch[1].tokens == std::vector<std::string>{
                               "[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]"}));

  const auto batch_pairs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"Hello, y'all!", "How are you \xF0\x9F\x98\x81 ?"},
          {"Hello to you too!", "I'm fine, thank you!"},
      },
      true);
  assert(batch_pairs.size() == 2);
  assert(batch_pairs[0].tokens == pair.tokens);
  assert(batch_pairs[0].type_ids == pair.type_ids);
  assert(batch_pairs[1].tokens.front() == "[CLS]");
  assert(batch_pairs[1].tokens.back() == "[SEP]");
  assert(std::find(
             batch_pairs[1].type_ids.begin(),
             batch_pairs[1].type_ids.end(),
             1) != batch_pairs[1].type_ids.end());
}

void test_quicktour_batch_padding(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_quicktour_tokenizer(data_dir, true, true);
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{
          "Hello, y'all!",
          "How are you \xF0\x9F\x98\x81 ?",
      },
      true);

  assert(outputs.size() == 2);
  assert((outputs[0].tokens == std::vector<std::string>{
                                "[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]"}));
  assert((outputs[0].attention_mask == std::vector<std::uint32_t>{
                                      1, 1, 1, 1, 1, 1, 1, 1}));
  assert((outputs[1].ids == std::vector<std::uint32_t>{
                                1,
                                7961,
                                5112,
                                6218,
                                0,
                                35,
                                2,
                                3,
                            }));
  assert((outputs[1].tokens == std::vector<std::string>{
                                "[CLS]",
                                "How",
                                "are",
                                "you",
                                "[UNK]",
                                "?",
                                "[SEP]",
                                "[PAD]",
                            }));
  assert((outputs[1].offsets == offsets({
                                  {0, 0},
                                  {0, 3},
                                  {4, 7},
                                  {8, 11},
                                  {12, 16},
                                  {17, 18},
                                  {0, 0},
                                  {0, 0},
                              })));
  assert(outputs[1].word_ids == word_ids({
                                    std::nullopt,
                                    0,
                                    1,
                                    2,
                                    3,
                                    4,
                                    std::nullopt,
                                    std::nullopt,
                                }));
  assert((outputs[1].type_ids == std::vector<std::uint32_t>{
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                }));
  assert((outputs[1].special_tokens_mask == std::vector<std::uint32_t>{
                                            1,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            1,
                                            1,
                                        }));
  assert((outputs[1].attention_mask == std::vector<std::uint32_t>{
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      1,
                                      0,
                                  }));
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_quicktour_initial_encode(data_dir);
  test_quicktour_template_single_pair_and_batch(data_dir);
  test_quicktour_batch_padding(data_dir);
  test_documentation_pipeline(data_dir);
  return 0;
}
