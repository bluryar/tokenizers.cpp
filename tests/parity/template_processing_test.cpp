#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

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

json wordlevel_tokenizer_json(json post_processor, json truncation = nullptr) {
  return {
      {"version", "1.0"},
      {"truncation", std::move(truncation)},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", {{"type", "WhitespaceSplit"}}},
      {"post_processor", std::move(post_processor)},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "WordLevel"},
           {"unk_token", "[UNK]"},
           {"vocab",
            {
                {"[UNK]", 0},
                {"one", 1},
                {"two", 2},
                {"three", 3},
                {"four", 4},
                {"five", 5},
                {"[CLS]", 101},
                {"[SEP]", 102},
                {"<PAIR>", 120},
                {"<X>", 201},
                {"<Y>", 202},
                {"<END>", 203},
            }},
       }},
  };
}

tokenizers_cpp::Tokenizer load_tokenizer(
    const std::string & name,
    const json & value) {
  const auto path = write_temp_tokenizer_json(name, value);
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

json broad_template_processor() {
  return {
      {"type", "TemplateProcessing"},
      {"single",
       json::array({
           {{"SpecialToken", {{"id", "<BUNDLE>"}, {"type_id", 7}}}},
           {{"Sequence", {{"id", "A"}, {"type_id", 3}}}},
           {{"SpecialToken", {{"id", "<END>"}, {"type_id", 4}}}},
       })},
      {"pair",
       json::array({
           {{"SpecialToken", {{"id", "<BUNDLE>"}, {"type_id", 9}}}},
           {{"Sequence", {{"id", "B"}, {"type_id", 5}}}},
           {{"SpecialToken", {{"id", "<PAIR>"}, {"type_id", 6}}}},
           {{"Sequence", {{"id", "A"}, {"type_id", 2}}}},
           {{"SpecialToken", {{"id", "<END>"}, {"type_id", 8}}}},
       })},
      {"special_tokens",
       {
           {"<BUNDLE>",
            {
                {"id", "<BUNDLE>"},
                {"ids", json::array({201, 202})},
                {"tokens", json::array({"<X>", "<Y>"})},
            }},
           {"<PAIR>",
            {
                {"id", "<PAIR>"},
                {"ids", json::array({120})},
                {"tokens", json::array({"<PAIR>"})},
            }},
           {"<END>",
            {
                {"id", "<END>"},
                {"ids", json::array({203})},
                {"tokens", json::array({"<END>"})},
            }},
       }},
  };
}

json bert_like_template_processor() {
  return {
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
                {"ids", json::array({101})},
                {"tokens", json::array({"[CLS]"})},
            }},
           {"[SEP]",
            {
                {"id", "[SEP]"},
                {"ids", json::array({102})},
                {"tokens", json::array({"[SEP]"})},
            }},
       }},
  };
}

json right_longest_first_truncation(std::uint32_t max_length) {
  return {
      {"direction", "Right"},
      {"max_length", max_length},
      {"strategy", "LongestFirst"},
      {"stride", 0},
  };
}

void test_multi_special_single_and_pair_templates() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_template_processing_multi_special.json",
      wordlevel_tokenizer_json(broad_template_processor()));

  const auto single = tokenizer.encode("one two", true);
  assert((single.ids == std::vector<std::uint32_t>{201, 202, 1, 2, 203}));
  assert((single.tokens == std::vector<std::string>{"<X>", "<Y>", "one", "two", "<END>"}));
  assert((single.type_ids == std::vector<std::uint32_t>{7, 7, 3, 3, 4}));
  assert((single.offsets == offsets({{0, 0}, {0, 0}, {0, 3}, {4, 7}, {0, 0}})));
  assert(single.word_ids == word_ids({std::nullopt, std::nullopt, 0U, 1U, std::nullopt}));
  assert((single.special_tokens_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 1}));
  assert((single.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));

  const auto pair = tokenizer.encode_pair("one", "two three", true);
  assert((pair.ids == std::vector<std::uint32_t>{201, 202, 2, 3, 120, 1, 203}));
  assert((pair.tokens == std::vector<std::string>{
                             "<X>", "<Y>", "two", "three", "<PAIR>", "one", "<END>"}));
  assert((pair.type_ids == std::vector<std::uint32_t>{9, 9, 5, 5, 6, 2, 8}));
  assert((pair.offsets == offsets({{0, 0}, {0, 0}, {0, 3}, {4, 9}, {0, 0}, {0, 3}, {0, 0}})));
  assert(pair.word_ids ==
      word_ids({std::nullopt, std::nullopt, 0U, 1U, std::nullopt, 0U, std::nullopt}));
  assert((pair.special_tokens_mask == std::vector<std::uint32_t>{1, 1, 0, 0, 1, 0, 1}));
  assert((pair.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1, 1, 1}));
}

void test_template_without_specials_keeps_sequence_order_and_type_ids() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_template_processing_no_specials.json",
      wordlevel_tokenizer_json(broad_template_processor()));

  const auto single = tokenizer.encode("one two", false);
  assert((single.ids == std::vector<std::uint32_t>{1, 2}));
  assert((single.type_ids == std::vector<std::uint32_t>{3, 3}));
  assert((single.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));

  const auto pair = tokenizer.encode_pair("one", "two three", false);
  assert((pair.ids == std::vector<std::uint32_t>{2, 3, 1}));
  assert((pair.tokens == std::vector<std::string>{"two", "three", "one"}));
  assert((pair.type_ids == std::vector<std::uint32_t>{5, 5, 2}));
  assert(pair.word_ids == word_ids({0U, 1U, 0U}));
  assert((pair.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0}));
}

void test_template_truncation_counts_multi_id_specials() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_template_processing_truncation_counts_specials.json",
      wordlevel_tokenizer_json(
          broad_template_processor(),
          right_longest_first_truncation(5)));

  const auto output = tokenizer.encode("one two three", true);
  assert((output.ids == std::vector<std::uint32_t>{201, 202, 1, 2, 203}));
  assert((output.type_ids == std::vector<std::uint32_t>{7, 7, 3, 3, 4}));
  assert(output.overflowing.size() == 1);
  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{201, 202, 3, 203}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{7, 7, 3, 4}));
  assert((output.overflowing[0].special_tokens_mask == std::vector<std::uint32_t>{1, 1, 0, 1}));
}

void test_pair_template_overflow_cross_product_is_nested() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_template_processing_pair_overflow_nested.json",
      wordlevel_tokenizer_json(
          bert_like_template_processor(),
          right_longest_first_truncation(7)));

  const auto output =
      tokenizer.encode_pair("one two three", "one two three", true);
  assert((output.ids == std::vector<std::uint32_t>{101, 1, 2, 102, 1, 2, 102}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1, 1}));
  assert(output.overflowing.size() == 3);

  const auto both_overflow_ids = std::vector<std::uint32_t>{101, 3, 102, 3, 102};
  const auto both_overflow_type_ids = std::vector<std::uint32_t>{0, 0, 0, 1, 1};

  assert((output.overflowing[0].ids == std::vector<std::uint32_t>{101, 3, 102, 1, 2, 102}));
  assert((output.overflowing[0].type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1, 1}));
  assert(output.overflowing[0].overflowing.size() == 1);
  assert(output.overflowing[0].overflowing[0].ids == both_overflow_ids);
  assert(output.overflowing[0].overflowing[0].type_ids == both_overflow_type_ids);

  assert(output.overflowing[1].ids == both_overflow_ids);
  assert(output.overflowing[1].type_ids == both_overflow_type_ids);

  assert((output.overflowing[2].ids == std::vector<std::uint32_t>{101, 1, 2, 102, 3, 102}));
  assert((output.overflowing[2].type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 1, 1}));
  assert(output.overflowing[2].overflowing.size() == 1);
  assert(output.overflowing[2].overflowing[0].ids == both_overflow_ids);
  assert(output.overflowing[2].overflowing[0].type_ids == both_overflow_type_ids);
}

}  // namespace

int main() {
  test_multi_special_single_and_pair_templates();
  test_template_without_specials_keeps_sequence_order_and_type_ids();
  test_template_truncation_counts_multi_id_specials();
  test_pair_template_overflow_cross_product_is_nested();
  return 0;
}
