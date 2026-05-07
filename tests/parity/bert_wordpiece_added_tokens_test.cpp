#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
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

json bert_added_mask() {
  return json::array({
      {
          {"id", 103},
          {"content", "[MASK]"},
          {"single_word", false},
          {"lstrip", false},
          {"rstrip", false},
          {"normalized", false},
          {"special", true},
      },
  });
}

json bert_tokenizer_json(
    const json & vocab,
    json added_tokens,
    std::uint32_t max_input_chars_per_word = 100,
    json normalizer = nullptr,
    json decoder = nullptr) {
  if (normalizer.is_null()) {
    normalizer = {
        {"type", "BertNormalizer"},
        {"clean_text", true},
        {"handle_chinese_chars", true},
        {"strip_accents", nullptr},
        {"lowercase", true},
    };
  }
  if (decoder.is_null()) {
    decoder = {{"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}};
  }

  return {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", std::move(added_tokens)},
      {"normalizer", std::move(normalizer)},
      {"pre_tokenizer", {{"type", "BertPreTokenizer"}}},
      {"post_processor",
       {
           {"type", "BertProcessing"},
           {"sep", json::array({"[SEP]", 102})},
           {"cls", json::array({"[CLS]", 101})},
       }},
      {"decoder", std::move(decoder)},
      {"model",
       {
           {"type", "WordPiece"},
           {"unk_token", "[UNK]"},
           {"continuing_subword_prefix", "##"},
           {"max_input_chars_per_word", max_input_chars_per_word},
           {"vocab", vocab},
       }},
  };
}

tokenizers_cpp::Tokenizer load_bert_wordpiece(
    const std::filesystem::path & data_dir,
    const std::string & name,
    json added_tokens,
    std::uint32_t max_input_chars_per_word = 100) {
  const auto vocab = read_wordpiece_vocab(data_dir / "bert-base-uncased-vocab.txt");
  const auto path = write_temp_tokenizer_json(
      name,
      bert_tokenizer_json(vocab, std::move(added_tokens), max_input_chars_per_word));
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_bert_wordpiece_with_vocab(
    const std::string & name,
    json vocab,
    json normalizer,
    json decoder = nullptr) {
  const auto path = write_temp_tokenizer_json(
      name,
      bert_tokenizer_json(
          std::move(vocab),
          json::array(),
          100,
          std::move(normalizer),
          std::move(decoder)));
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_real_bert_with_truncation_padding(
    const std::filesystem::path & data_dir,
    std::uint32_t max_length = 5) {
  auto tokenizer_json = read_json(data_dir / "bert-wiki.json");
  tokenizer_json["truncation"] = {
      {"direction", "Right"},
      {"max_length", max_length},
      {"strategy", "LongestFirst"},
      {"stride", 0},
  };
  tokenizer_json["padding"] = {
      {"strategy", {{"Fixed", 8}}},
      {"direction", "Right"},
      {"pad_to_multiple_of", nullptr},
      {"pad_id", 3},
      {"pad_type_id", 0},
      {"pad_token", "[PAD]"},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_real_bert_truncation_padding.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void assert_real_bert_raw_single_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            8396,
                            2,
                            3,
                            3,
                            3,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "world",
                               "test",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {6, 11},
                              {12, 16},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                2,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                22453,
                                7118,
                                2,
                                3,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.tokens == std::vector<std::string>{
                                "[CLS]",
                                "tok",
                                "##en",
                                "[SEP]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {17, 20},
                                {20, 22},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  3,
                                  3,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));
}

void assert_real_bert_pretokenized_single_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            8396,
                            2,
                            3,
                            3,
                            3,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "world",
                               "test",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 5},
                              {0, 4},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                2,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                22453,
                                7118,
                                2,
                                3,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.tokens == std::vector<std::string>{
                                "[CLS]",
                                "tok",
                                "##en",
                                "[SEP]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {0, 3},
                                {3, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  3,
                                  3,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));
}

void assert_real_bert_short_single(const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{1, 27462, 2, 3, 3, 3, 3, 3}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 0, 0, 0}));
  assert(output.overflowing.empty());
}

void assert_real_bert_raw_pair_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            2,
                            7601,
                            2,
                            3,
                            3,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "world",
                               "[SEP]",
                               "world",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {6, 11},
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                8396,
                                2,
                                7601,
                                2,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.tokens == std::vector<std::string>{
                                "[CLS]",
                                "test",
                                "[SEP]",
                                "world",
                                "[SEP]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {12, 16},
                                {0, 0},
                                {0, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  2,
                                  std::nullopt,
                                  0,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

void assert_real_bert_pretokenized_pair_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            2,
                            7601,
                            2,
                            3,
                            3,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "world",
                               "[SEP]",
                               "world",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 5},
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                8396,
                                2,
                                7601,
                                2,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.tokens == std::vector<std::string>{
                                "[CLS]",
                                "test",
                                "[SEP]",
                                "world",
                                "[SEP]",
                                "[PAD]",
                                "[PAD]",
                                "[PAD]",
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {0, 4},
                                {0, 0},
                                {0, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  2,
                                  std::nullopt,
                                  0,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

void assert_real_bert_short_pair(const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{1, 27462, 2, 8396, 2, 3, 3, 3}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "[SEP]",
                               "test",
                               "[SEP]",
                               "[PAD]",
                               "[PAD]",
                               "[PAD]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 4},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
  assert(output.overflowing.empty());
}

void assert_real_bert_raw_single_char_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            8396,
                            2,
                            3,
                            3,
                            3,
                        }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {6, 11},
                              {12, 16},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 1,
                                 2,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                22453,
                                7118,
                                2,
                                3,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {17, 20},
                                {20, 22},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  3,
                                  3,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));
}

void assert_real_bert_pretokenized_single_char_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            8396,
                            2,
                            3,
                            3,
                            3,
                        }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 5},
                              {0, 4},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 1,
                                 2,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{
                                1,
                                22453,
                                7118,
                                2,
                                3,
                                3,
                                3,
                                3,
                            }));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {0, 3},
                                {3, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  3,
                                  3,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));
}

void assert_real_bert_short_single_char(const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{1, 27462, 2, 3, 3, 3, 3, 3}));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 0, 0, 0}));
  assert(output.overflowing.empty());
}

void assert_real_bert_raw_pair_char_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            2,
                            7601,
                            2,
                            3,
                            3,
                        }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {6, 11},
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 0, 0}));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 1,
                                 std::nullopt,
                                 0,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{1, 8396, 2, 7601, 2, 3, 3, 3}));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {12, 16},
                                {0, 0},
                                {0, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert((overflow.word_ids == word_ids({
                                   std::nullopt,
                                   2,
                                   std::nullopt,
                                   0,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                               })));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

void assert_real_bert_pretokenized_pair_char_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            27462,
                            7601,
                            2,
                            7601,
                            2,
                            3,
                            3,
                        }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 5},
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 0, 0}));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 1,
                                 std::nullopt,
                                 0,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{1, 8396, 2, 7601, 2, 3, 3, 3}));
  assert((overflow.offsets == offsets({
                                {0, 0},
                                {0, 4},
                                {0, 0},
                                {0, 5},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            })));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert((overflow.word_ids == word_ids({
                                   std::nullopt,
                                   2,
                                   std::nullopt,
                                   0,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                               })));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

void assert_real_bert_short_pair_char(const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{1, 27462, 2, 8396, 2, 3, 3, 3}));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 4},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          })));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 0, 0, 0}));
  assert((output.word_ids == word_ids({
                                 std::nullopt,
                                 0,
                                 std::nullopt,
                                 0,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                                 std::nullopt,
                             })));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 0, 0, 0}));
  assert(output.overflowing.empty());
}

void test_split_on_added_tokens_bert(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_split_on_added_tokens_bert.json",
      bert_added_mask());
  const auto output = tokenizer.encode("Yesterday I saw a [MASK] far away", false);

  assert((output.ids == std::vector<std::uint32_t>{
                            7483,
                            1045,
                            2387,
                            1037,
                            103,
                            2521,
                            2185,
                        }));
  assert((output.offsets == offsets({
                              {0, 9},
                              {10, 11},
                              {12, 15},
                              {16, 17},
                              {18, 24},
                              {25, 28},
                              {29, 33},
                          })));
  assert((output.tokens == std::vector<std::string>{
                               "yesterday",
                               "i",
                               "saw",
                               "a",
                               "[MASK]",
                               "far",
                               "away",
                           }));
  assert((output.word_ids == std::vector<std::optional<std::uint32_t>>{
                                 0U,
                                 1U,
                                 2U,
                                 3U,
                                 4U,
                                 5U,
                                 6U,
                             }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                        }));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1}));
  assert(tokenizer.token_to_id("[MASK]").value() == 103);
  assert(tokenizer.id_to_token(103).value() == "[MASK]");
}

void test_wordpiece_subword_and_unknown_offsets(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_wordpiece_subword_unknown.json",
      json::array());
  const std::string input =
      std::string("unaffable tokenizers ") + "\xF0\x9F\xA4\x97";
  const auto output = tokenizer.encode(input, false);

  assert((output.ids == std::vector<std::uint32_t>{
                            14477,
                            20961,
                            3468,
                            19204,
                            17629,
                            2015,
                            100,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "una",
                               "##ffa",
                               "##ble",
                               "token",
                               "##izer",
                               "##s",
                               "[UNK]",
                           }));
  assert((output.offsets == offsets({
                              {0, 3},
                              {3, 6},
                              {6, 9},
                              {10, 15},
                              {15, 19},
                              {19, 20},
                              {21, 25},
                          })));
  assert((output.word_ids == word_ids({0U, 0U, 0U, 1U, 1U, 1U, 2U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1}));
}

void test_wordpiece_max_input_chars_unknown(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_wordpiece_max_chars.json",
      json::array(),
      4);
  const auto output = tokenizer.encode("hello", false);

  assert((output.ids == std::vector<std::uint32_t>{100}));
  assert((output.tokens == std::vector<std::string>{"[UNK]"}));
  assert((output.offsets == offsets({{0, 5}})));
  assert((output.word_ids == word_ids({0U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1}));
}

void test_bert_unicode_normalizer_with_real_vocab(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_bert_unicode_normalizer.json",
      json::array());

  const auto output = tokenizer.encode(
      "H\xC3\xA9llo\t\xE4\xB8\x96 TEST",
      false);

  assert((output.ids == std::vector<std::uint32_t>{7592, 1745, 3231}));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               "\xE4\xB8\x96",
                               "test",
                           }));
  assert((output.offsets == offsets({{0, 6}, {7, 10}, {11, 15}})));
  assert((output.word_ids == word_ids({0U, 1U, 2U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));

  const auto removed_control = tokenizer.encode(
      "he\xE2\x80\x8Bllo",
      false);
  assert((removed_control.ids == std::vector<std::uint32_t>{7592}));
  assert((removed_control.tokens == std::vector<std::string>{"hello"}));
  assert((removed_control.offsets == offsets({{0, 8}})));
  assert((removed_control.word_ids == word_ids({0U})));
}

void test_bert_unicode_lowercase_subset() {
  const auto tokenizer = load_bert_wordpiece_with_vocab(
      "tokenizers_cpp_bert_unicode_lowercase_subset.json",
      {
          {"[UNK]", 0},
          {"\xCE\xB1\xCE\xB2\xCE\xB3", 1},
          {"\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", 2},
      },
      {
          {"type", "BertNormalizer"},
          {"clean_text", true},
          {"handle_chinese_chars", true},
          {"strip_accents", false},
          {"lowercase", true},
      });

  const auto output = tokenizer.encode(
      "\xCE\x91\xCE\x92\xCE\x93 \xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2",
      false);

  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{
                               "\xCE\xB1\xCE\xB2\xCE\xB3",
                               "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82",
                           }));
  assert((output.offsets == offsets({{0, 6}, {7, 19}})));
  assert((output.word_ids == word_ids({0U, 1U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_bert_icu_unicode_normalizer() {
  const auto tokenizer = load_bert_wordpiece_with_vocab(
      "tokenizers_cpp_bert_icu_unicode_normalizer.json",
      {
          {"[UNK]", 0},
          {"\xCE\xB1", 1},
          {"hello", 2},
      },
      {
          {"type", "BertNormalizer"},
          {"clean_text", true},
          {"handle_chinese_chars", true},
          {"strip_accents", nullptr},
          {"lowercase", true},
      });

  const auto output = tokenizer.encode(
      "\xE1\xBC\x88 he\xEE\x80\x80llo",
      false);

  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"\xCE\xB1", "hello"}));
  assert((output.offsets == offsets({{0, 3}, {4, 12}})));
  assert((output.word_ids == word_ids({0U, 1U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(output.ids, false) == "\xCE\xB1 hello");
}

void test_bert_unicode_punctuation_with_real_vocab(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_bert_unicode_punctuation.json",
      json::array());

  const auto output = tokenizer.encode(
      "Hello\xEF\xBC\x8Cworld\xEF\xBC\x81\xC2\xBFhello\xE2\x80\x94world\xE3\x80\x82",
      false);

  assert((output.ids == std::vector<std::uint32_t>{
                            7592,
                            1989,
                            2088,
                            1986,
                            1094,
                            7592,
                            1517,
                            2088,
                            1636,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               "\xEF\xBC\x8C",
                               "world",
                               "\xEF\xBC\x81",
                               "\xC2\xBF",
                               "hello",
                               "\xE2\x80\x94",
                               "world",
                               "\xE3\x80\x82",
                           }));
  assert((output.offsets == offsets({
                              {0, 5},
                              {5, 8},
                              {8, 13},
                              {13, 16},
                              {16, 18},
                              {18, 23},
                              {23, 26},
                              {26, 31},
                              {31, 34},
                          })));
  assert((output.word_ids == word_ids({0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0}));
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
                                        }));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

void test_bert_icu_punctuation_category() {
  const auto tokenizer = load_bert_wordpiece_with_vocab(
      "tokenizers_cpp_bert_icu_punctuation_category.json",
      {
          {"[UNK]", 0},
          {"hello", 1},
          {"\xF0\x90\x84\x80", 2},
          {"world", 3},
      },
      {
          {"type", "BertNormalizer"},
          {"clean_text", false},
          {"handle_chinese_chars", false},
          {"strip_accents", false},
          {"lowercase", false},
      },
      {
          {"type", "WordPiece"},
          {"prefix", "##"},
          {"cleanup", false},
      });

  const auto output = tokenizer.encode(
      "hello" "\xF0\x90\x84\x80" "world",
      false);

  assert((output.ids == std::vector<std::uint32_t>{1, 2, 3}));
  assert((output.tokens == std::vector<std::string>{
                               "hello",
                               "\xF0\x90\x84\x80",
                               "world",
                           }));
  assert((output.offsets == offsets({{0, 5}, {5, 9}, {9, 14}})));
  assert((output.word_ids == word_ids({0U, 1U, 2U})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) ==
      "hello " "\xF0\x90\x84\x80" " world");
}

void test_wordpiece_decoder_cleanup() {
  const auto tokenizer = load_bert_wordpiece_with_vocab(
      "tokenizers_cpp_wordpiece_decoder_cleanup.json",
      {
          {"[UNK]", 0},
          {"i", 1},
          {"'m", 2},
          {"hello", 3},
          {"##s", 4},
          {"!", 5},
          {"?", 6},
          {",", 7},
      },
      {
          {"type", "BertNormalizer"},
          {"clean_text", false},
          {"handle_chinese_chars", false},
          {"strip_accents", false},
          {"lowercase", false},
      });

  assert(tokenizer.decode({1, 2, 3, 4, 5, 6, 7}, false) == "i'm hellos!?,");
}

void test_wordpiece_decoder_without_cleanup_keeps_initial_prefix() {
  const auto tokenizer = load_bert_wordpiece_with_vocab(
      "tokenizers_cpp_wordpiece_decoder_no_cleanup.json",
      {
          {"[UNK]", 0},
          {"##uelo", 1},
          {"Ara", 2},
          {"##\xC3\xBAj", 3},
          {"##o", 4},
          {"No", 5},
          {"##guera", 6},
      },
      {
          {"type", "BertNormalizer"},
          {"clean_text", false},
          {"handle_chinese_chars", false},
          {"strip_accents", false},
          {"lowercase", false},
      },
      {
          {"type", "WordPiece"},
          {"prefix", "##"},
          {"cleanup", false},
      });

  assert(tokenizer.decode({1, 2, 3, 4, 5, 6}, false) == "##uelo Ara\xC3\xBAjo Noguera");
}

void test_documentation_pipeline_bert(const std::filesystem::path & data_dir) {
  auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "bert-wiki.json");

  const auto output = tokenizer.encode(
      "Welcome to the \xF0\x9F\xA4\x97 Tokenizers library.",
      true);

  assert((output.ids == std::vector<std::uint32_t>{
                            1,
                            18263,
                            7128,
                            7108,
                            0,
                            22453,
                            27107,
                            12800,
                            4068,
                            11046,
                            18,
                            2,
                        }));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "welcome",
                               "to",
                               "the",
                               "[UNK]",
                               "tok",
                               "##eni",
                               "##zer",
                               "##s",
                               "library",
                               ".",
                               "[SEP]",
                           }));
  assert((output.offsets == offsets({
                              {0, 0},
                              {0, 7},
                              {8, 10},
                              {11, 14},
                              {15, 19},
                              {20, 23},
                              {23, 26},
                              {26, 29},
                              {29, 30},
                              {31, 38},
                              {38, 39},
                              {0, 0},
                          })));
  assert((output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                2,
                                3,
                                4,
                                4,
                                4,
                                4,
                                5,
                                6,
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
                                  }));

  assert(tokenizer.decode(output.ids, true) ==
      "welcome to the tok ##eni ##zer ##s library .");
  tokenizer.with_wordpiece_decoder();
  assert(tokenizer.decode(output.ids, true) == "welcome to the tokenizers library.");
}

void test_bert_processing_single_sequence(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_bert_processing_single.json",
      json::array());
  const auto output = tokenizer.encode("unaffable", true);

  assert((output.ids == std::vector<std::uint32_t>{101, 14477, 20961, 3468, 102}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "una",
                               "##ffa",
                               "##ble",
                               "[SEP]",
                           }));
  assert((output.offsets == offsets({{0, 0}, {0, 3}, {3, 6}, {6, 9}, {0, 0}})));
  assert((output.word_ids == word_ids({std::nullopt, 0U, 0U, 0U, std::nullopt})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
}

void test_bert_processing_pair_sequence(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_bert_wordpiece(
      data_dir,
      "tokenizers_cpp_bert_processing_pair.json",
      json::array());

  const auto without_specials = tokenizer.encode_pair("hello", "world", false);
  assert((without_specials.ids == std::vector<std::uint32_t>{7592, 2088}));
  assert((without_specials.tokens == std::vector<std::string>{"hello", "world"}));
  assert((without_specials.offsets == offsets({{0, 5}, {0, 5}})));
  assert((without_specials.word_ids == word_ids({0U, 0U})));
  assert((without_specials.type_ids == std::vector<std::uint32_t>{0, 1}));
  assert((without_specials.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));

  const auto output = tokenizer.encode_pair("hello", "world", true);
  assert((output.ids == std::vector<std::uint32_t>{101, 7592, 102, 2088, 102}));
  assert((output.tokens == std::vector<std::string>{
                               "[CLS]",
                               "hello",
                               "[SEP]",
                               "world",
                               "[SEP]",
                           }));
  assert((output.offsets == offsets({{0, 0}, {0, 5}, {0, 0}, {0, 5}, {0, 0}})));
  assert((output.word_ids == word_ids({std::nullopt, 0U, std::nullopt, 0U, std::nullopt})));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
}

void test_real_bert_json_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello world test token", true);
  assert_real_bert_raw_single_overflow(output);
}

void test_real_bert_batch_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"Hello world test token", "Hello"},
      true);
  assert(outputs.size() == 2);
  assert_real_bert_raw_single_overflow(outputs[0]);
  assert_real_bert_short_single(outputs[1]);
}

void test_real_bert_pretokenized_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode(
      std::vector<std::string>{"Hello", "world", "test", "token"},
      true);
  assert_real_bert_pretokenized_single_overflow(output);
}

void test_real_bert_batch_pretokenized_truncation_padding_preserves_word_offsets(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::vector<std::string>>{
          {"Hello", "world", "test", "token"},
          {"Hello"},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_pretokenized_single_overflow(outputs[0]);
  assert_real_bert_short_single(outputs[1]);
}

void test_real_bert_pair_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output =
      tokenizer.encode_pair("Hello world test", "world", true);
  assert_real_bert_raw_pair_overflow(output);
}

void test_real_bert_batch_pair_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"Hello world test", "world"},
          {"Hello", "test"},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_raw_pair_overflow(outputs[0]);
  assert_real_bert_short_pair(outputs[1]);
}

void test_real_bert_pretokenized_pair_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair(
      std::vector<std::string>{"Hello", "world", "test"},
      std::vector<std::string>{"world"},
      true);
  assert_real_bert_pretokenized_pair_overflow(output);
}

void test_real_bert_batch_pretokenized_pair_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>{
          {{"Hello", "world", "test"}, {"world"}},
          {{"Hello"}, {"test"}},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_pretokenized_pair_overflow(outputs[0]);
  assert_real_bert_short_pair(outputs[1]);
}

void test_real_bert_char_offsets_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output =
      tokenizer.encode_char_offsets("H\xC3\xA9llo world test token", true);
  assert_real_bert_raw_single_char_overflow(output);
}

void test_real_bert_batch_char_offsets_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_char_offsets(
      std::vector<std::string>{"H\xC3\xA9llo world test token", "H\xC3\xA9llo"},
      true);
  assert(outputs.size() == 2);
  assert_real_bert_raw_single_char_overflow(outputs[0]);
  assert_real_bert_short_single_char(outputs[1]);
}

void test_real_bert_pretokenized_char_offsets_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_char_offsets(
      std::vector<std::string>{"H\xC3\xA9llo", "world", "test", "token"},
      true);
  assert_real_bert_pretokenized_single_char_overflow(output);
}

void test_real_bert_batch_pretokenized_char_offsets_truncation_padding_preserves_word_offsets(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_char_offsets(
      std::vector<std::vector<std::string>>{
          {"H\xC3\xA9llo", "world", "test", "token"},
          {"H\xC3\xA9llo"},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_pretokenized_single_char_overflow(outputs[0]);
  assert_real_bert_short_single_char(outputs[1]);
}

void test_real_bert_pair_char_offsets_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair_char_offsets(
      "H\xC3\xA9llo world test",
      "world",
      true);
  assert_real_bert_raw_pair_char_overflow(output);
}

void test_real_bert_batch_pair_char_offsets_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_pairs_char_offsets(
      std::vector<std::pair<std::string, std::string>>{
          {"H\xC3\xA9llo world test", "world"},
          {"H\xC3\xA9llo", "test"},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_raw_pair_char_overflow(outputs[0]);
  assert_real_bert_short_pair_char(outputs[1]);
}

void test_real_bert_pretokenized_pair_char_offsets_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair_char_offsets(
      std::vector<std::string>{"H\xC3\xA9llo", "world", "test"},
      std::vector<std::string>{"world"},
      true);
  assert_real_bert_pretokenized_pair_char_overflow(output);
}

void test_real_bert_batch_pretokenized_pair_char_offsets_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_pairs_char_offsets(
      std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>{
          {{"H\xC3\xA9llo", "world", "test"}, {"world"}},
          {{"H\xC3\xA9llo"}, {"test"}},
      },
      true);
  assert(outputs.size() == 2);
  assert_real_bert_pretokenized_pair_char_overflow(outputs[0]);
  assert_real_bert_short_pair_char(outputs[1]);
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto real_bert_truncating_tokenizer =
      load_real_bert_with_truncation_padding(data_dir);
  const auto real_bert_pair_truncating_tokenizer =
      load_real_bert_with_truncation_padding(data_dir, 6);

  test_split_on_added_tokens_bert(data_dir);
  test_wordpiece_subword_and_unknown_offsets(data_dir);
  test_wordpiece_max_input_chars_unknown(data_dir);
  test_bert_unicode_normalizer_with_real_vocab(data_dir);
  test_bert_unicode_lowercase_subset();
  test_bert_icu_unicode_normalizer();
  test_bert_unicode_punctuation_with_real_vocab(data_dir);
  test_bert_icu_punctuation_category();
  test_wordpiece_decoder_cleanup();
  test_wordpiece_decoder_without_cleanup_keeps_initial_prefix();
  test_documentation_pipeline_bert(data_dir);
  test_bert_processing_single_sequence(data_dir);
  test_bert_processing_pair_sequence(data_dir);
  test_real_bert_json_truncation_padding_overflow(real_bert_truncating_tokenizer);
  test_real_bert_batch_truncation_padding_preserves_order(real_bert_truncating_tokenizer);
  test_real_bert_pretokenized_truncation_padding_overflow(
      real_bert_truncating_tokenizer);
  test_real_bert_batch_pretokenized_truncation_padding_preserves_word_offsets(
      real_bert_truncating_tokenizer);
  test_real_bert_pair_truncation_padding_overflow(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_batch_pair_truncation_padding_preserves_order(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_pretokenized_pair_truncation_padding_overflow(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_batch_pretokenized_pair_truncation_padding_preserves_order(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_char_offsets_truncation_padding_overflow(
      real_bert_truncating_tokenizer);
  test_real_bert_batch_char_offsets_truncation_padding_preserves_order(
      real_bert_truncating_tokenizer);
  test_real_bert_pretokenized_char_offsets_truncation_padding_overflow(
      real_bert_truncating_tokenizer);
  test_real_bert_batch_pretokenized_char_offsets_truncation_padding_preserves_word_offsets(
      real_bert_truncating_tokenizer);
  test_real_bert_pair_char_offsets_truncation_padding_overflow(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_batch_pair_char_offsets_truncation_padding_preserves_order(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_pretokenized_pair_char_offsets_truncation_padding_overflow(
      real_bert_pair_truncating_tokenizer);
  test_real_bert_batch_pretokenized_pair_char_offsets_truncation_padding_preserves_order(
      real_bert_pair_truncating_tokenizer);
  return 0;
}
