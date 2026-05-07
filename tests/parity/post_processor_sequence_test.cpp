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

tokenizers_cpp::Tokenizer load_byte_level_template_sequence_tokenizer(
    const std::filesystem::path & data_dir,
    bool with_truncation_padding = false,
    std::uint32_t truncation_max_length = 3) {
  auto tokenizer_json = read_json(data_dir / "tokenizer.json");
  tokenizer_json["pre_tokenizer"] = {
      {"type", "ByteLevel"},
      {"add_prefix_space", true},
      {"trim_offsets", true},
      {"use_regex", true},
  };
  tokenizer_json["decoder"] = {
      {"type", "ByteLevel"},
      {"add_prefix_space", true},
      {"trim_offsets", true},
      {"use_regex", true},
  };
  tokenizer_json["post_processor"] = {
      {"type", "Sequence"},
      {"processors",
       json::array({
           {
               {"type", "ByteLevel"},
               {"add_prefix_space", true},
               {"trim_offsets", true},
               {"use_regex", true},
           },
           {
               {"type", "TemplateProcessing"},
               {"single",
                json::array({
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                    {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                })},
               {"pair",
                json::array({
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                    {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                    {{"SpecialToken", {{"id", "<s>"}, {"type_id", 1}}}},
                    {{"Sequence", {{"id", "B"}, {"type_id", 1}}}},
                })},
               {"special_tokens",
                {
                    {"<s>",
                     {
                         {"id", "<s>"},
                         {"ids", json::array({0})},
                         {"tokens", json::array({"<s>"})},
                     }},
                }},
           },
       })},
  };
  if (with_truncation_padding) {
    tokenizer_json["truncation"] = {
        {"direction", "Right"},
        {"max_length", truncation_max_length},
        {"strategy", "LongestFirst"},
        {"stride", 0},
    };
    tokenizer_json["padding"] = {
        {"strategy", {{"Fixed", 5}}},
        {"direction", "Right"},
        {"pad_to_multiple_of", nullptr},
        {"pad_id", 1},
        {"pad_type_id", 0},
        {"pad_token", "<pad>"},
    };
  }

  const auto path = write_temp_tokenizer_json(
      with_truncation_padding
          ? "tokenizers_cpp_post_processor_sequence_truncation_padding.json"
          : "tokenizers_cpp_post_processor_sequence.json",
      tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void assert_sequence_pretokenized_single_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{0, 90, 81, 1, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "\xC4\xA0the",
                              "\xC4\xA0" "a",
                              "<pad>",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 3}, {0, 1}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, 1, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{0, 90, 1, 1, 1}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<s>",
                                "\xC4\xA0the",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {0, 3}, {0, 0}, {0, 0}, {0, 0}}));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(overflow.word_ids ==
      word_ids({std::nullopt, 2, std::nullopt, std::nullopt, std::nullopt}));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
}

void assert_sequence_pretokenized_pair_overflow(
    const tokenizers_cpp::Encoding & output) {
  assert((output.ids == std::vector<std::uint32_t>{0, 90, 0, 90, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "\xC4\xA0the",
                              "<s>",
                              "\xC4\xA0the",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 3}, {0, 0}, {0, 3}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0}));

  assert(output.overflowing.size() == 2);
  const auto & first_overflow = output.overflowing[0];
  assert((first_overflow.ids == std::vector<std::uint32_t>{0, 81, 0, 90, 1}));
  assert((first_overflow.tokens == std::vector<std::string>{
                                     "<s>",
                                     "\xC4\xA0" "a",
                                     "<s>",
                                     "\xC4\xA0the",
                                     "<pad>",
                                 }));
  assert((first_overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                     {0, 0}, {0, 1}, {0, 0}, {0, 3}, {0, 0}}));
  assert((first_overflow.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0}));
  assert(first_overflow.word_ids ==
      word_ids({std::nullopt, 1, std::nullopt, 0, std::nullopt}));
  assert((first_overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((first_overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0}));

  const auto & second_overflow = output.overflowing[1];
  assert((second_overflow.ids == std::vector<std::uint32_t>{0, 90, 0, 90, 1}));
  assert((second_overflow.tokens == std::vector<std::string>{
                                      "<s>",
                                      "\xC4\xA0the",
                                      "<s>",
                                      "\xC4\xA0the",
                                      "<pad>",
                                  }));
  assert((second_overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 0}, {0, 3}, {0, 0}, {0, 3}, {0, 0}}));
  assert((second_overflow.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0}));
  assert(second_overflow.word_ids ==
      word_ids({std::nullopt, 2, std::nullopt, 0, std::nullopt}));
  assert((second_overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((second_overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0}));
}

void test_sequence_byte_level_template_single(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("the a", true);
  assert((output.ids == std::vector<std::uint32_t>{0, 90, 81}));
  assert((output.tokens == std::vector<std::string>{"<s>", "\xC4\xA0the", "\xC4\xA0" "a"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 3}, {4, 5}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert(output.word_ids == word_ids({std::nullopt, 0, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
  assert(tokenizer.decode(output.ids, true) == " the a");
  assert(tokenizer.decode(output.ids, false) == "<s> the a");
}

void test_sequence_byte_level_template_strip_normalizer(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("  the a  ", true);
  assert((output.ids == std::vector<std::uint32_t>{0, 90, 81}));
  assert((output.tokens == std::vector<std::string>{"<s>", "\xC4\xA0the", "\xC4\xA0" "a"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {2, 5}, {6, 7}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert(output.word_ids == word_ids({std::nullopt, 0, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1}));
  assert(tokenizer.decode(output.ids, true) == " the a");
  assert(tokenizer.decode(output.ids, false) == "<s> the a");
}

void test_sequence_byte_level_template_pair(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair("the", "a", true);
  assert((output.ids == std::vector<std::uint32_t>{0, 90, 0, 81}));
  assert((output.tokens == std::vector<std::string>{"<s>", "\xC4\xA0the", "<s>", "\xC4\xA0" "a"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 3}, {0, 0}, {0, 1}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1}));
  assert(output.word_ids == word_ids({std::nullopt, 0, std::nullopt, 0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_sequence_byte_level_template_truncation_padding(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("the a the", true);

  assert((output.ids == std::vector<std::uint32_t>{0, 90, 81, 1, 1}));
  assert((output.tokens == std::vector<std::string>{
                              "<s>",
                              "\xC4\xA0the",
                              "\xC4\xA0" "a",
                              "<pad>",
                              "<pad>",
                          }));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 0}, {0, 3}, {4, 5}, {0, 0}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(output.word_ids ==
      word_ids({std::nullopt, 0, 1, std::nullopt, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 0, 0}));

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert((overflow.ids == std::vector<std::uint32_t>{0, 90, 1, 1, 1}));
  assert((overflow.tokens == std::vector<std::string>{
                                "<s>",
                                "\xC4\xA0the",
                                "<pad>",
                                "<pad>",
                                "<pad>",
                            }));
  assert((overflow.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {6, 9}, {0, 0}, {0, 0}, {0, 0}}));
  assert((overflow.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(overflow.word_ids ==
      word_ids({std::nullopt, 2, std::nullopt, std::nullopt, std::nullopt}));
  assert((overflow.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1}));
  assert((overflow.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
}

void test_sequence_byte_level_template_pretokenized_truncation_padding(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output =
      tokenizer.encode(std::vector<std::string>{"the", "a", "the"}, true);
  assert_sequence_pretokenized_single_overflow(output);
}

void test_sequence_byte_level_template_batch_pretokenized_truncation_padding(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::vector<std::string>>{
          {"the", "a", "the"},
          {"the"},
      },
      true);
  assert(outputs.size() == 2);
  assert_sequence_pretokenized_single_overflow(outputs[0]);

  const auto & second = outputs[1];
  assert((second.ids == std::vector<std::uint32_t>{0, 90, 1, 1, 1}));
  assert((second.tokens == std::vector<std::string>{
                               "<s>",
                               "\xC4\xA0the",
                               "<pad>",
                               "<pad>",
                               "<pad>",
                           }));
  assert((second.offsets == std::vector<tokenizers_cpp::Offset>{
                               {0, 0}, {0, 3}, {0, 0}, {0, 0}, {0, 0}}));
  assert((second.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(second.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, std::nullopt}));
  assert((second.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 1, 1}));
  assert((second.attention_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 0}));
  assert(second.overflowing.empty());
}

void test_sequence_byte_level_template_pair_pretokenized_truncation_padding(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair(
      std::vector<std::string>{"the", "a", "the"},
      std::vector<std::string>{"the"},
      true);
  assert_sequence_pretokenized_pair_overflow(output);
}

void test_sequence_byte_level_template_batch_pair_pretokenized_truncation_padding(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>{
          {{"the", "a", "the"}, {"the"}},
          {{"the"}, {"a"}},
      },
      true);
  assert(outputs.size() == 2);
  assert_sequence_pretokenized_pair_overflow(outputs[0]);

  const auto & second = outputs[1];
  assert((second.ids == std::vector<std::uint32_t>{0, 90, 0, 81, 1}));
  assert((second.tokens == std::vector<std::string>{
                               "<s>",
                               "\xC4\xA0the",
                               "<s>",
                               "\xC4\xA0" "a",
                               "<pad>",
                           }));
  assert((second.offsets == std::vector<tokenizers_cpp::Offset>{
                               {0, 0}, {0, 3}, {0, 0}, {0, 1}, {0, 0}}));
  assert((second.type_ids == std::vector<std::uint32_t>{0, 0, 1, 1, 0}));
  assert(second.word_ids ==
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt}));
  assert((second.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((second.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 0}));
  assert(second.overflowing.empty());
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto tokenizer = load_byte_level_template_sequence_tokenizer(data_dir);
  const auto truncating_tokenizer =
      load_byte_level_template_sequence_tokenizer(data_dir, true);
  const auto pair_truncating_tokenizer =
      load_byte_level_template_sequence_tokenizer(data_dir, true, 4);

  test_sequence_byte_level_template_single(tokenizer);
  test_sequence_byte_level_template_strip_normalizer(tokenizer);
  test_sequence_byte_level_template_pair(tokenizer);
  test_sequence_byte_level_template_truncation_padding(truncating_tokenizer);
  test_sequence_byte_level_template_pretokenized_truncation_padding(
      truncating_tokenizer);
  test_sequence_byte_level_template_batch_pretokenized_truncation_padding(
      truncating_tokenizer);
  test_sequence_byte_level_template_pair_pretokenized_truncation_padding(
      pair_truncating_tokenizer);
  test_sequence_byte_level_template_batch_pair_pretokenized_truncation_padding(
      pair_truncating_tokenizer);
  return 0;
}
