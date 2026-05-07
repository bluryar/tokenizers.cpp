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

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

std::filesystem::path write_temp_tokenizer_json(const std::string & name, const json & value) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << value;
  return path;
}

tokenizers_cpp::Tokenizer load_wordlevel_roberta_with_truncation() {
  const json tokenizer_json = {
      {"version", "1.0"},
      {"truncation",
       {
           {"max_length", 6},
           {"strategy", "OnlyFirst"},
           {"stride", 0},
           {"direction", "Right"},
       }},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", {{"type", "WhitespaceSplit"}}},
      {"post_processor",
       {
           {"type", "RobertaProcessing"},
           {"sep", {"</s>", 2}},
           {"cls", {"<s>", 0}},
           {"trim_offsets", true},
           {"add_prefix_space", false},
       }},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "WordLevel"},
           {"vocab",
            {
                {"<s>", 0},
                {"<pad>", 1},
                {"</s>", 2},
                {"<unk>", 3},
                {"one", 4},
                {"two", 5},
                {"three", 6},
                {"four", 7},
            }},
           {"unk_token", "<unk>"},
       }},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_roberta_wordlevel_truncation.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_real_roberta_with_truncation_padding(
    const std::filesystem::path & data_dir,
    std::uint32_t max_length = 4) {
  auto tokenizer_json = read_json(data_dir / "roberta.json");
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
      {"pad_id", 1},
      {"pad_type_id", 0},
      {"pad_token", "<pad>"},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_roberta_real_truncation_padding.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_roberta_json_single(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto without_specials = tokenizer.encode("Hello world", false);
  assert((without_specials.ids == std::vector<std::uint32_t>{31414, 232}));
  assert((without_specials.tokens == std::vector<std::string>{"Hello", "\xC4\xA0world"}));
  assert((without_specials.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 11}}));
  assert((without_specials.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(without_specials.word_ids == word_ids({0, 1}));
  assert((without_specials.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((without_specials.attention_mask == std::vector<std::uint32_t>{1, 1}));

  const auto with_specials = tokenizer.encode("Hello world", true);
  assert((with_specials.ids == std::vector<std::uint32_t>{0, 31414, 232, 2}));
  assert((with_specials.tokens == std::vector<std::string>{
                                     "<s>", "Hello", "\xC4\xA0world", "</s>"}));
  assert((with_specials.offsets == std::vector<tokenizers_cpp::Offset>{
                                     {0, 0}, {0, 5}, {6, 11}, {0, 0}}));
  assert((with_specials.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(with_specials.word_ids == word_ids({std::nullopt, 0, 1, std::nullopt}));
  assert((with_specials.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1}));
  assert((with_specials.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
  assert(tokenizer.decode(with_specials.ids, true) == "Hello world");
  assert(tokenizer.decode(with_specials.ids, false) == "<s>Hello world</s>");
}

void test_roberta_json_pair(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto without_specials = tokenizer.encode_pair("Hello", "world", false);
  assert((without_specials.ids == std::vector<std::uint32_t>{31414, 8331}));
  assert((without_specials.tokens == std::vector<std::string>{"Hello", "world"}));
  assert((without_specials.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {0, 5}}));
  assert((without_specials.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(without_specials.word_ids == word_ids({0, 0}));
  assert((without_specials.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((without_specials.attention_mask == std::vector<std::uint32_t>{1, 1}));

  const auto with_specials = tokenizer.encode_pair("Hello", "world", true);
  assert((with_specials.ids == std::vector<std::uint32_t>{0, 31414, 2, 2, 8331, 2}));
  assert((with_specials.tokens == std::vector<std::string>{
                                  "<s>", "Hello", "</s>", "</s>", "world", "</s>"}));
  assert((with_specials.offsets == std::vector<tokenizers_cpp::Offset>{
                                  {0, 0}, {0, 5}, {0, 0}, {0, 0}, {0, 5}, {0, 0}}));
  assert((with_specials.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(with_specials.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, 0, std::nullopt}));
  assert((with_specials.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1}));
  assert((with_specials.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1}));
}

void test_roberta_json_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello world test", true);

  assert((output.ids == std::vector<std::uint32_t>{0, 31414, 232, 2, 1, 1, 1, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "Hello",
                              "\xC4\xA0world",
                              "</s>",
                              "<pad>",
                              "<pad>",
                              "<pad>",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0},
                              {0, 5},
                              {6, 11},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{0, 1296, 2, 1, 1, 1, 1, 1}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<s>",
                                "\xC4\xA0test",
                                "</s>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0},
                                {12, 16},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            }));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  2,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 0, 0, 0}));
}

void assert_roberta_pretokenized_truncated_padded_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{0, 31414, 8331, 2, 1, 1, 1, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "Hello",
                              "world",
                              "</s>",
                              "<pad>",
                              "<pad>",
                              "<pad>",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0},
                              {0, 5},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                1,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 0, 1, 1, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0, 0, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{0, 21959, 2, 1, 1, 1, 1, 1}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<s>",
                                "test",
                                "</s>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0},
                                {0, 4},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                            }));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(overflow.word_ids == word_ids({
                                  std::nullopt,
                                  2,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                              }));
  assert((overflow.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 0, 0, 0}));
}

void test_roberta_pretokenized_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output =
      tokenizer.encode(std::vector<std::string>{"Hello", "world", "test"}, true);
  assert_roberta_pretokenized_truncated_padded_overflow(output);
}

void test_roberta_batch_pretokenized_truncation_padding_preserves_word_offsets(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::vector<std::string>>{{"Hello", "world", "test"}, {"Hello"}},
      true);

  assert(outputs.size() == 2);
  assert_roberta_pretokenized_truncated_padded_overflow(outputs[0]);

  const auto & short_output = outputs[1];
  assert((short_output.ids == std::vector<std::uint32_t>{0, 31414, 2, 1, 1, 1, 1, 1}));
  assert((short_output.tokens == std::vector<std::string>{
                                      "<s>",
                                      "Hello",
                                      "</s>",
                                      "<pad>",
                                      "<pad>",
                                      "<pad>",
                                      "<pad>",
                                      "<pad>",
                                  }));
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0},
                                      {0, 5},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                  }));
  assert((short_output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(short_output.word_ids == word_ids({
                                      std::nullopt,
                                      0,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                  }));
  assert((short_output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 1, 1, 1, 1}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0, 0, 0, 0}));
  assert(short_output.overflowing.empty());
}

void assert_roberta_pretokenized_pair_truncated_padded_overflows(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{0, 31414, 2, 2, 8331, 2, 1, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "Hello",
                              "</s>",
                              "</s>",
                              "world",
                              "</s>",
                              "<pad>",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 5},
                              {0, 0},
                              {0, 0},
                              {0, 0},
                          }));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(output.word_ids == word_ids({
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                0,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                            }));
  assert((output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 2);
  const auto & world = output.overflowing[0];
  assert((world.ids == std::vector<std::uint32_t>{0, 8331, 2, 2, 8331, 2, 1, 1}));
  assert((world.tokens == std::vector<std::string>{
                             "<s>",
                             "world",
                             "</s>",
                             "</s>",
                             "world",
                             "</s>",
                             "<pad>",
                             "<pad>",
                         }));
  assert((world.offsets == std::vector<tokenizers_cpp::Offset>{
                             {0, 0},
                             {0, 5},
                             {0, 0},
                             {0, 0},
                             {0, 5},
                             {0, 0},
                             {0, 0},
                             {0, 0},
                         }));
  assert((world.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(world.word_ids == word_ids({
                               std::nullopt,
                               1,
                               std::nullopt,
                               std::nullopt,
                               0,
                               std::nullopt,
                               std::nullopt,
                               std::nullopt,
                           }));
  assert((world.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1, 1, 1}));
  assert((world.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  const auto & test = output.overflowing[1];
  assert((test.ids == std::vector<std::uint32_t>{0, 21959, 2, 2, 8331, 2, 1, 1}));
  assert((test.tokens == std::vector<std::string>{
                            "<s>",
                            "test",
                            "</s>",
                            "</s>",
                            "world",
                            "</s>",
                            "<pad>",
                            "<pad>",
                        }));
  assert((test.offsets == std::vector<tokenizers_cpp::Offset>{
                            {0, 0},
                            {0, 4},
                            {0, 0},
                            {0, 0},
                            {0, 5},
                            {0, 0},
                            {0, 0},
                            {0, 0},
                        }));
  assert((test.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(test.word_ids == word_ids({
                              std::nullopt,
                              2,
                              std::nullopt,
                              std::nullopt,
                              0,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                          }));
  assert((test.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1, 1, 1}));
  assert((test.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));
}

void test_roberta_pretokenized_pair_truncation_padding_overflow(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair(
      std::vector<std::string>{"Hello", "world", "test"},
      std::vector<std::string>{"world"},
      true);
  assert_roberta_pretokenized_pair_truncated_padded_overflows(output);
}

void test_roberta_batch_pretokenized_pair_truncation_padding_preserves_order(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  using PretokenizedPair =
      std::pair<std::vector<std::string>, std::vector<std::string>>;
  const std::vector<PretokenizedPair> inputs = {
      {{"Hello", "world", "test"}, {"world"}},
      {{"Hello"}, {"test"}},
  };
  const auto outputs = tokenizer.encode_batch_pairs(inputs, true);

  assert(outputs.size() == 2);
  assert_roberta_pretokenized_pair_truncated_padded_overflows(outputs[0]);

  const auto & short_output = outputs[1];
  assert((short_output.ids == std::vector<std::uint32_t>{0, 31414, 2, 2, 21959, 2, 1, 1}));
  assert((short_output.tokens == std::vector<std::string>{
                                      "<s>",
                                      "Hello",
                                      "</s>",
                                      "</s>",
                                      "test",
                                      "</s>",
                                      "<pad>",
                                      "<pad>",
                                  }));
  assert((short_output.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0},
                                      {0, 5},
                                      {0, 0},
                                      {0, 0},
                                      {0, 4},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                  }));
  assert((short_output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0, 0, 0}));
  assert(short_output.word_ids == word_ids({
                                      std::nullopt,
                                      0,
                                      std::nullopt,
                                      std::nullopt,
                                      0,
                                      std::nullopt,
                                      std::nullopt,
                                      std::nullopt,
                                  }));
  assert((short_output.special_tokens_mask ==
          std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1, 1, 1}));
  assert((short_output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 0, 0}));
  assert(short_output.overflowing.empty());
}

void test_roberta_pair_truncation_counts_four_special_tokens() {
  const auto tokenizer = load_wordlevel_roberta_with_truncation();
  const auto output = tokenizer.encode_pair("one two three", "four", true);

  assert((output.ids == std::vector<std::uint32_t>{0, 4, 2, 2, 7, 2}));
  assert((output.tokens == std::vector<std::string>{
                         "<s>", "one", "</s>", "</s>", "four", "</s>"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                         {0, 0}, {0, 3}, {0, 0}, {0, 0}, {0, 4}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, 0, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1}));
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");
  const auto single_truncating_tokenizer =
      load_real_roberta_with_truncation_padding(data_dir);
  const auto pair_truncating_tokenizer =
      load_real_roberta_with_truncation_padding(data_dir, 6);

  test_roberta_json_single(tokenizer);
  test_roberta_json_pair(tokenizer);
  test_roberta_json_truncation_padding_overflow(single_truncating_tokenizer);
  test_roberta_pretokenized_truncation_padding_overflow(single_truncating_tokenizer);
  test_roberta_batch_pretokenized_truncation_padding_preserves_word_offsets(
      single_truncating_tokenizer);
  test_roberta_pretokenized_pair_truncation_padding_overflow(pair_truncating_tokenizer);
  test_roberta_batch_pretokenized_pair_truncation_padding_preserves_order(
      pair_truncating_tokenizer);
  test_roberta_pair_truncation_counts_four_special_tokens();
  return 0;
}
