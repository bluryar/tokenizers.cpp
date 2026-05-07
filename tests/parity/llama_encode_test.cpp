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

std::filesystem::path write_temp_tokenizer_json(const std::string & name, const json & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

tokenizers_cpp::Tokenizer load_llama_with_truncation_padding(
    const std::filesystem::path & data_dir) {
  auto tokenizer_json = read_json(data_dir / "llama-3-tokenizer.json");
  tokenizer_json["truncation"] = {
      {"direction", "Right"},
      {"max_length", 4},
      {"strategy", "LongestFirst"},
      {"stride", 0},
  };
  tokenizer_json["padding"] = {
      {"strategy", {{"Fixed", 6}}},
      {"direction", "Right"},
      {"pad_to_multiple_of", nullptr},
      {"pad_id", 128004},
      {"pad_type_id", 0},
      {"pad_token", "<|finetune_right_pad_id|>"},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_llama_truncation_padding.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_llama_encode_without_specials(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hey! how is this token: ", false);
  assert((output.ids == std::vector<std::uint32_t>{
                            19182, 0, 1268, 374, 420, 4037, 25, 220}));
  assert((output.tokens == std::vector<std::string>{
                               "Hey",
                               "!",
                               "\xC4\xA0how",
                               "\xC4\xA0is",
                               "\xC4\xA0this",
                               "\xC4\xA0token",
                               ":",
                               "\xC4\xA0"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 3},
                                {3, 4},
                                {4, 8},
                                {8, 11},
                                {11, 16},
                                {16, 22},
                                {22, 23},
                                {23, 24}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 1, 2, 3, 4, 5, 6, 7}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == "Hey! how is this token: ");
}

void test_llama_encode_with_template_processing(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello, world!", true);
  assert((output.ids == std::vector<std::uint32_t>{128000, 9906, 11, 1917, 0}));
  assert((output.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             "Hello",
                             ",",
                             "\xC4\xA0world",
                             "!"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {5, 6}, {6, 12}, {12, 13}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({std::nullopt, 0, 1, 2, 3}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, true) == "Hello, world!");
}

void test_llama_split_digits_into_three_digit_chunks(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("abc 1234", false);
  assert((output.ids == std::vector<std::uint32_t>{13997, 220, 4513, 19}));
  assert((output.tokens == std::vector<std::string>{"abc", "\xC4\xA0", "123", "4"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 3}, {3, 4}, {4, 7}, {7, 8}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 1, 2, 3}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_llama_contractions(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("I'm you're they'll don't", false);
  assert((output.ids == std::vector<std::uint32_t>{
                            40, 2846, 499, 2351, 814, 3358, 1541, 956}));
  assert((output.tokens == std::vector<std::string>{
                               "I",
                               "'m",
                               "\xC4\xA0you",
                               "'re",
                               "\xC4\xA0they",
                               "'ll",
                               "\xC4\xA0" "don",
                               "'t"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 1},
                                {1, 3},
                                {3, 7},
                                {7, 10},
                                {10, 15},
                                {15, 18},
                                {18, 22},
                                {22, 24}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 1, 2, 3, 4, 5, 6, 7}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == "I'm you're they'll don't");
}

void test_llama_newline_splits(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello\nworld\r\n  next", false);
  assert((output.ids == std::vector<std::uint32_t>{9906, 198, 14957, 319, 220, 1828}));
  assert((output.tokens == std::vector<std::string>{
                             "Hello",
                             "\xC4\x8A",
                             "world",
                             "\xC4\x8D\xC4\x8A",
                             "\xC4\xA0",
                             "\xC4\xA0next"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 5}, {5, 6}, {6, 11}, {11, 13}, {13, 14}, {14, 19}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 1, 2, 3, 4, 5}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == "Hello\nworld\r\n  next");
}

void test_llama_unicode_letter_number_splits(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::string input =
      "caf\xC3\xA9 \xE6\x9D\xB1\xE4\xBA\xAC \xEF\xBC\x91\xEF\xBC\x92" "3";
  const auto output = tokenizer.encode(input, false);
  assert((output.ids == std::vector<std::uint32_t>{
                            936, 59958, 119109, 220, 20713, 25963, 18}));
  assert((output.tokens == std::vector<std::string>{
                               "ca",
                               "f\xC3\x83\xC2\xA9",
                               "\xC4\xA0\xC3\xA6\xC4\xBF\xC2\xB1\xC3\xA4\xC2\xBA\xC2\xAC",
                               "\xC4\xA0",
                               "\xC3\xAF\xC2\xBC\xC4\xB3",
                               "\xC3\xAF\xC2\xBC\xC4\xB4",
                               "3"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 2},
                                {2, 5},
                                {5, 12},
                                {12, 13},
                                {13, 16},
                                {16, 19},
                                {19, 20}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 0, 1, 2, 3, 3, 3}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == input);
}

void test_llama_icu_unicode_letter_number_splits(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::string input =
      "A\xE1\x88\xB4" "B \xE2\x85\xAB" "3";
  const auto output = tokenizer.encode(input, false);
  assert((output.ids == std::vector<std::uint32_t>{
                            32, 157, 230, 112, 33, 220, 71567, 104, 18}));
  assert((output.tokens == std::vector<std::string>{
                               "A",
                               "\xC3\xA1",
                               "\xC4\xAA",
                               "\xC2\xB4",
                               "B",
                               "\xC4\xA0",
                               "\xC3\xA2\xC4\xA7",
                               "\xC2\xAB",
                               "3"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 1},
                                {1, 4},
                                {1, 4},
                                {1, 4},
                                {4, 5},
                                {5, 6},
                                {6, 9},
                                {6, 9},
                                {9, 10}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({0, 0, 0, 0, 0, 1, 2, 2, 2}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, false) == input);
}

void test_llama_pair_template_processing(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair("Hello", "world", true);
  assert((output.ids == std::vector<std::uint32_t>{128000, 9906, 128000, 14957}));
  assert((output.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             "Hello",
                             "<|begin_of_text|>",
                             "world"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {0, 0}, {0, 5}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1}));
  assert(output.word_ids == word_ids({std::nullopt, 0, std::nullopt, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void assert_llama_truncated_padded_overflow(const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{128000, 9906, 11, 1917, 128004, 128004}));
  assert((output.tokens == std::vector<std::string>{
                              "<|begin_of_text|>",
                              "Hello",
                              ",",
                              "\xC4\xA0world",
                              "<|finetune_right_pad_id|>",
                              "<|finetune_right_pad_id|>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {5, 6}, {6, 12}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, 1, 2, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{128000, 0, 128004, 128004, 128004, 128004}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<|begin_of_text|>",
                                "!",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {12, 13}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids ==
      word_ids({std::nullopt, 3, std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));
}

void test_llama_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello, world!", true);
  assert_llama_truncated_padded_overflow(output);
}

void test_llama_batch_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs =
      tokenizer.encode_batch(std::vector<std::string>{"Hello, world!", "Hello"}, true);

  assert(outputs.size() == 2);
  assert_llama_truncated_padded_overflow(outputs[0]);

  const auto & short_output = outputs[1];
  assert((short_output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 128004, 128004, 128004, 128004}));
  assert((short_output.tokens == std::vector<std::string>{
                                      "<|begin_of_text|>",
                                      "Hello",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                  }));
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {0, 5}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((short_output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(short_output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
  assert((short_output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));
  assert(short_output.overflowing.empty());
}

void test_llama_batch_char_offsets_truncation_padding_unicode(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::string unicode_input =
      "caf\xC3\xA9 \xE6\x9D\xB1\xE4\xBA\xAC \xEF\xBC\x91\xEF\xBC\x92" "3";
  const auto outputs = tokenizer.encode_batch_char_offsets(
      std::vector<std::string>{unicode_input, "Hello"},
      true);

  assert(outputs.size() == 2);
  const auto & output = outputs[0];
  assert((output.ids ==
          std::vector<std::uint32_t>{128000, 936, 59958, 119109, 128004, 128004}));
  assert((output.tokens == std::vector<std::string>{
                              "<|begin_of_text|>",
                              "ca",
                              "f\xC3\x83\xC2\xA9",
                              "\xC4\xA0\xC3\xA6\xC4\xBF\xC2\xB1\xC3\xA4\xC2\xBA\xC2\xAC",
                              "<|finetune_right_pad_id|>",
                              "<|finetune_right_pad_id|>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 2}, {2, 4}, {4, 7}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, 0, 1, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 2);
  const auto & first_overflow = output.overflowing[0];
  assert((first_overflow.ids ==
          std::vector<std::uint32_t>{128000, 220, 20713, 25963, 128004, 128004}));
  assert((first_overflow.tokens == std::vector<std::string>{
                                     "<|begin_of_text|>",
                                     "\xC4\xA0",
                                     "\xC3\xAF\xC2\xBC\xC4\xB3",
                                     "\xC3\xAF\xC2\xBC\xC4\xB4",
                                     "<|finetune_right_pad_id|>",
                                     "<|finetune_right_pad_id|>",
                                 }));
  assert((first_overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                     {0, 0}, {7, 8}, {8, 9}, {9, 10}, {0, 0}, {0, 0}}));
  assert((first_overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(first_overflow.word_ids ==
      word_ids({std::nullopt, 2, 3, 3, std::nullopt, std::nullopt}));
  assert((first_overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1}));
  assert((first_overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  const auto & second_overflow = output.overflowing[1];
  assert((second_overflow.ids ==
          std::vector<std::uint32_t>{128000, 18, 128004, 128004, 128004, 128004}));
  assert((second_overflow.tokens == std::vector<std::string>{
                                      "<|begin_of_text|>",
                                      "3",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                  }));
  assert((second_overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {10, 11}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((second_overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(second_overflow.word_ids ==
      word_ids({std::nullopt, 3, std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
  assert((second_overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1}));
  assert((second_overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));

  const auto & short_output = outputs[1];
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {0, 5}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));
  assert(short_output.overflowing.empty());
}

void assert_llama_pretokenized_truncated_padded_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 11, 14957, 128004, 128004}));
  assert((output.tokens == std::vector<std::string>{
                              "<|begin_of_text|>",
                              "Hello",
                              ",",
                              "world",
                              "<|finetune_right_pad_id|>",
                              "<|finetune_right_pad_id|>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {5, 6}, {0, 5}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, 0, 1, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids ==
          std::vector<std::uint32_t>{128000, 0, 128004, 128004, 128004, 128004}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<|begin_of_text|>",
                                "!",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                                "<|finetune_right_pad_id|>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {5, 6}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids ==
      word_ids({std::nullopt, 1, std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));
}

void test_llama_batch_pretokenized_truncation_padding_preserves_word_offsets(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::vector<std::vector<std::string>> inputs = {
      {"Hello,", "world!"},
      {"Hello"},
  };
  const auto outputs = tokenizer.encode_batch(inputs, true);

  assert(outputs.size() == 2);
  assert_llama_pretokenized_truncated_padded_overflow(outputs[0]);

  const auto & short_output = outputs[1];
  assert((short_output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 128004, 128004, 128004, 128004}));
  assert((short_output.tokens == std::vector<std::string>{
                                      "<|begin_of_text|>",
                                      "Hello",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                  }));
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {0, 5}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}));
  assert((short_output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(short_output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt}));
  assert((short_output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0, 0}));
  assert(short_output.overflowing.empty());
}

void assert_llama_pair_truncated_padded_overflows(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 128000, 14957, 128004, 128004}));
  assert((output.tokens == std::vector<std::string>{
                              "<|begin_of_text|>",
                              "Hello",
                              "<|begin_of_text|>",
                              "world",
                              "<|finetune_right_pad_id|>",
                              "<|finetune_right_pad_id|>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 3);
  const auto & comma = output.overflowing[0];
  assert((comma.ids ==
          std::vector<std::uint32_t>{128000, 11, 128000, 14957, 128004, 128004}));
  assert((comma.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             ",",
                             "<|begin_of_text|>",
                             "world",
                             "<|finetune_right_pad_id|>",
                             "<|finetune_right_pad_id|>",
                         }));
  assert((comma.offsets == std::vector<tokenizers_cpp::Offset>{
                             {0, 0}, {5, 6}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((comma.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(comma.word_ids ==
      word_ids({std::nullopt, 1, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((comma.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((comma.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  const auto & world = output.overflowing[1];
  assert((world.ids ==
          std::vector<std::uint32_t>{128000, 1917, 128000, 14957, 128004, 128004}));
  assert((world.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             "\xC4\xA0world",
                             "<|begin_of_text|>",
                             "world",
                             "<|finetune_right_pad_id|>",
                             "<|finetune_right_pad_id|>",
                         }));
  assert((world.offsets == std::vector<tokenizers_cpp::Offset>{
                             {0, 0}, {6, 12}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((world.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(world.word_ids ==
      word_ids({std::nullopt, 2, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((world.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((world.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  const auto & bang = output.overflowing[2];
  assert((bang.ids ==
          std::vector<std::uint32_t>{128000, 0, 128000, 14957, 128004, 128004}));
  assert((bang.tokens == std::vector<std::string>{
                            "<|begin_of_text|>",
                            "!",
                            "<|begin_of_text|>",
                            "world",
                            "<|finetune_right_pad_id|>",
                            "<|finetune_right_pad_id|>",
                        }));
  assert((bang.offsets == std::vector<tokenizers_cpp::Offset>{
                            {0, 0}, {12, 13}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((bang.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(bang.word_ids ==
      word_ids({std::nullopt, 3, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((bang.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((bang.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));
}

void test_llama_pair_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair("Hello, world!", "world", true);
  assert_llama_pair_truncated_padded_overflows(output);
}

void assert_llama_short_pair_truncated_padded(
    const tokenizers_cpp::Encoding & short_output) {
  assert((short_output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 128000, 1985, 128004, 128004}));
  assert((short_output.tokens == std::vector<std::string>{
                                      "<|begin_of_text|>",
                                      "Hello",
                                      "<|begin_of_text|>",
                                      "test",
                                      "<|finetune_right_pad_id|>",
                                      "<|finetune_right_pad_id|>",
                                  }));
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {0, 5}, {0, 0}, {0, 4}, {0, 0}, {0, 0}}));
  assert((short_output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(short_output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((short_output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));
  assert(short_output.overflowing.empty());
}

void test_llama_batch_pair_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::vector<std::pair<std::string, std::string>> inputs = {
      {"Hello, world!", "world"},
      {"Hello", "test"},
  };
  const auto outputs = tokenizer.encode_batch_pairs(inputs, true);

  assert(outputs.size() == 2);
  assert_llama_pair_truncated_padded_overflows(outputs[0]);
  assert_llama_short_pair_truncated_padded(outputs[1]);
}

void test_llama_batch_pair_char_offsets_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const std::vector<std::pair<std::string, std::string>> inputs = {
      {"Hello, world!", "world"},
      {"Hello", "test"},
  };
  const auto outputs = tokenizer.encode_batch_pairs_char_offsets(inputs, true);

  assert(outputs.size() == 2);
  assert_llama_pair_truncated_padded_overflows(outputs[0]);
  assert_llama_short_pair_truncated_padded(outputs[1]);
}

void assert_llama_pretokenized_pair_truncated_padded_overflows(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids ==
          std::vector<std::uint32_t>{128000, 9906, 128000, 14957, 128004, 128004}));
  assert((output.tokens == std::vector<std::string>{
                              "<|begin_of_text|>",
                              "Hello",
                              "<|begin_of_text|>",
                              "world",
                              "<|finetune_right_pad_id|>",
                              "<|finetune_right_pad_id|>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 5}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 3);
  const auto & comma = output.overflowing[0];
  assert((comma.ids ==
          std::vector<std::uint32_t>{128000, 11, 128000, 14957, 128004, 128004}));
  assert((comma.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             ",",
                             "<|begin_of_text|>",
                             "world",
                             "<|finetune_right_pad_id|>",
                             "<|finetune_right_pad_id|>",
                         }));
  assert((comma.offsets == std::vector<tokenizers_cpp::Offset>{
                             {0, 0}, {5, 6}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((comma.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(comma.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((comma.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((comma.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  const auto & world = output.overflowing[1];
  assert((world.ids ==
          std::vector<std::uint32_t>{128000, 14957, 128000, 14957, 128004, 128004}));
  assert((world.tokens == std::vector<std::string>{
                             "<|begin_of_text|>",
                             "world",
                             "<|begin_of_text|>",
                             "world",
                             "<|finetune_right_pad_id|>",
                             "<|finetune_right_pad_id|>",
                         }));
  assert((world.offsets == std::vector<tokenizers_cpp::Offset>{
                             {0, 0}, {0, 5}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((world.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(world.word_ids ==
      word_ids({std::nullopt, 1, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((world.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((world.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));

  const auto & bang = output.overflowing[2];
  assert((bang.ids ==
          std::vector<std::uint32_t>{128000, 0, 128000, 14957, 128004, 128004}));
  assert((bang.tokens == std::vector<std::string>{
                            "<|begin_of_text|>",
                            "!",
                            "<|begin_of_text|>",
                            "world",
                            "<|finetune_right_pad_id|>",
                            "<|finetune_right_pad_id|>",
                        }));
  assert((bang.offsets == std::vector<tokenizers_cpp::Offset>{
                            {0, 0}, {5, 6}, {0, 0}, {0, 5}, {0, 0}, {0, 0}}));
  assert((bang.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0, 0}));
  assert(bang.word_ids ==
      word_ids({std::nullopt, 1, std::nullopt, 0, std::nullopt, std::nullopt}));
  assert((bang.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1, 1}));
  assert((bang.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0}));
}

void test_llama_batch_pretokenized_pair_truncation_padding_preserves_word_offsets(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  using PretokenizedPair =
      std::pair<std::vector<std::string>, std::vector<std::string>>;
  const std::vector<PretokenizedPair> inputs = {
      {{"Hello,", "world!"}, {"world"}},
      {{"Hello"}, {"test"}},
  };
  const auto outputs = tokenizer.encode_batch_pairs(inputs, true);

  assert(outputs.size() == 2);
  assert_llama_pretokenized_pair_truncated_padded_overflows(outputs[0]);
  assert_llama_short_pair_truncated_padded(outputs[1]);
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");

  test_llama_encode_without_specials(tokenizer);
  test_llama_encode_with_template_processing(tokenizer);
  test_llama_split_digits_into_three_digit_chunks(tokenizer);
  test_llama_contractions(tokenizer);
  test_llama_newline_splits(tokenizer);
  test_llama_unicode_letter_number_splits(tokenizer);
  test_llama_icu_unicode_letter_number_splits(tokenizer);
  test_llama_pair_template_processing(tokenizer);

  const auto truncating_tokenizer = load_llama_with_truncation_padding(data_dir);
  test_llama_truncation_padding_overflow(truncating_tokenizer);
  test_llama_batch_truncation_padding_preserves_order(truncating_tokenizer);
  test_llama_batch_char_offsets_truncation_padding_unicode(truncating_tokenizer);
  test_llama_batch_pretokenized_truncation_padding_preserves_word_offsets(
      truncating_tokenizer);
  test_llama_pair_truncation_padding_overflow(truncating_tokenizer);
  test_llama_batch_pair_truncation_padding_preserves_order(truncating_tokenizer);
  test_llama_batch_pair_char_offsets_truncation_padding_preserves_order(
      truncating_tokenizer);
  test_llama_batch_pretokenized_pair_truncation_padding_preserves_word_offsets(
      truncating_tokenizer);
  return 0;
}
