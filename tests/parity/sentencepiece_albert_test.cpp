#include "tokenizers_cpp/tokenizer.hpp"

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

json unigram_vocab(std::initializer_list<std::string> tokens) {
  json vocab = json::array({json::array({"<unk>", 0.0})});
  double score = 1.0;
  for (const auto & token : tokens) {
    vocab.push_back(json::array({token, score}));
    score += 1.0;
  }
  return vocab;
}

tokenizers_cpp::Tokenizer load_unigram_sentencepiece_tokenizer(
    const std::string & name,
    const json & normalizer,
    const json & vocab) {
  const json tokenizer_json = {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", normalizer},
      {"pre_tokenizer",
       {
           {"type", "Sequence"},
           {"pretokenizers", json::array({{{"type", "WhitespaceSplit"}}})},
       }},
      {"post_processor", nullptr},
      {"decoder", nullptr},
      {"model",
       {
           {"type", "Unigram"},
           {"unk_id", 0},
           {"byte_fallback", false},
           {"vocab", vocab},
       }},
  };

  const auto path = write_temp_tokenizer_json(name, tokenizer_json);
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_albert_single_without_specials(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello world", false);
  assert((output.ids == std::vector<std::uint32_t>{10975, 126}));
  assert((output.tokens == std::vector<std::string>{"\xE2\x96\x81hello", "\xE2\x96\x81world"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{{0, 5}, {6, 11}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(output.word_ids == word_ids({0, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_albert_single_with_template_processing(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode("Hello world", true);
  assert((output.ids == std::vector<std::uint32_t>{2, 10975, 126, 3}));
  assert((output.tokens == std::vector<std::string>{
                            "[CLS]", "\xE2\x96\x81hello", "\xE2\x96\x81world", "[SEP]"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {0, 5}, {6, 11}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(output.word_ids == word_ids({std::nullopt, 0, 1, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
  assert(tokenizer.decode(output.ids, true) == "hello world");
}

void test_albert_nfkd_strip_accents(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto accented = tokenizer.encode("H\xC3\xA9ll\xC3\xB2 h\xC3\xB4w are \xC3\xBC?", false);
  assert((accented.ids == std::vector<std::uint32_t>{10975, 184, 50, 287, 60}));
  assert((accented.tokens == std::vector<std::string>{
                                "\xE2\x96\x81hello",
                                "\xE2\x96\x81how",
                                "\xE2\x96\x81" "are",
                                "\xE2\x96\x81u",
                                "?"}));
  assert((accented.offsets == std::vector<tokenizers_cpp::Offset>{
                                  {0, 7}, {8, 12}, {13, 16}, {17, 19}, {19, 20}}));
  assert((accented.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert(accented.word_ids == word_ids({0, 1, 2, 3, 3}));
  assert((accented.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0, 0}));
  assert((accented.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
  assert(tokenizer.decode(accented.ids, true) == "hello how are u?");

  const auto compatibility = tokenizer.encode("\xE1\xBA\xAD" "\xE2\x80\xA6", false);
  assert((compatibility.ids == std::vector<std::uint32_t>{21, 9, 9, 9}));
  assert((compatibility.tokens == std::vector<std::string>{
                                     "\xE2\x96\x81" "a", ".", ".", "."}));
  assert((compatibility.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 3}, {3, 6}, {3, 6}, {3, 6}}));
  assert((compatibility.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(compatibility.word_ids == word_ids({0, 0, 0, 0}));
  assert((compatibility.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((compatibility.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
  assert(tokenizer.decode(compatibility.ids, true) == "a...");

  const auto thai = tokenizer.encode(
      "\xE0\xB8\xB3" "\xE0\xB8\x99" "\xE0\xB9\x89" "\xE0\xB8\xB3"
      "3"
      "\xE0\xB8\xA5" "\xE0\xB8\xB3",
      false);
  assert((thai.ids == std::vector<std::uint32_t>{13, 1, 240, 1}));
  assert((thai.tokens == std::vector<std::string>{
                            "\xE2\x96\x81",
                            "\xE0\xB8\xB2" "\xE0\xB8\x99" "\xE0\xB8\xB2",
                            "3",
                            "\xE0\xB8\xA5" "\xE0\xB8\xB2"}));
  assert((thai.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 3}, {0, 12}, {12, 13}, {13, 19}}));
  assert((thai.type_ids == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(thai.word_ids == word_ids({0, 0, 0, 0}));
  assert((thai.special_tokens_mask == std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((thai.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1}));
}

void test_albert_precompiled_charsmap(const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto joined = tokenizer.encode("A\xE2\x80\x8D" "B", false);
  assert((joined.ids == std::vector<std::uint32_t>{21, 334}));
  assert((joined.tokens == std::vector<std::string>{
                             "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((joined.offsets == std::vector<tokenizers_cpp::Offset>{
                              {0, 1}, {4, 5}}));
  assert((joined.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(joined.word_ids == word_ids({0, 1}));
  assert((joined.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((joined.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(joined.ids, true) == "a b");

  const auto non_joined = tokenizer.encode("A\xE2\x80\x8C" "B", false);
  assert((non_joined.ids == std::vector<std::uint32_t>{21, 334}));
  assert((non_joined.tokens == std::vector<std::string>{
                                 "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((non_joined.offsets == std::vector<tokenizers_cpp::Offset>{
                                   {0, 1}, {4, 5}}));
  assert((non_joined.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(non_joined.word_ids == word_ids({0, 1}));
  assert((non_joined.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((non_joined.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(non_joined.ids, true) == "a b");

  const auto joined_words = tokenizer.encode("hello\xE2\x80\x8D" "world", false);
  assert((joined_words.ids == std::vector<std::uint32_t>{10975, 126}));
  assert((joined_words.tokens == std::vector<std::string>{
                                   "\xE2\x96\x81hello", "\xE2\x96\x81world"}));
  assert((joined_words.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 5}, {8, 13}}));
  assert((joined_words.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(joined_words.word_ids == word_ids({0, 1}));
  assert((joined_words.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((joined_words.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(joined_words.ids, true) == "hello world");

  const auto record_separator = tokenizer.encode("hello\x1E" "world", false);
  assert((record_separator.ids == std::vector<std::uint32_t>{10975, 4423}));
  assert((record_separator.tokens == std::vector<std::string>{
                                      "\xE2\x96\x81hello", "world"}));
  assert((record_separator.offsets == std::vector<tokenizers_cpp::Offset>{
                                           {0, 5}, {6, 11}}));
  assert((record_separator.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(record_separator.word_ids == word_ids({0, 0}));
  assert((
      record_separator.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((record_separator.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(record_separator.ids, true) == "helloworld");

  const auto expansion_then_removal = tokenizer.encode(
      "\xE2\x84\xA2\x1E" "g",
      false);
  assert((
      expansion_then_removal.ids ==
      std::vector<std::uint32_t>{13, 38, 11984}));
  assert((expansion_then_removal.tokens == std::vector<std::string>{
                                                "\xE2\x96\x81", "t", "mg"}));
  assert((expansion_then_removal.offsets == std::vector<tokenizers_cpp::Offset>{
                                                   {0, 3}, {0, 3}, {0, 5}}));
  assert((
      expansion_then_removal.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert(expansion_then_removal.word_ids == word_ids({0, 0, 0}));
  assert((
      expansion_then_removal.special_tokens_mask ==
      std::vector<std::uint32_t>{0, 0, 0}));
  assert((
      expansion_then_removal.attention_mask ==
      std::vector<std::uint32_t>{1, 1, 1}));
  assert(tokenizer.decode(expansion_then_removal.ids, true) == "tmg");

  const auto zero_width_space = tokenizer.encode("A\xE2\x80\x8B" "B", false);
  assert((zero_width_space.ids == std::vector<std::uint32_t>{21, 334}));
  assert((zero_width_space.tokens == std::vector<std::string>{
                                      "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((zero_width_space.offsets == std::vector<tokenizers_cpp::Offset>{
                                       {0, 1}, {4, 5}}));
  assert((zero_width_space.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(zero_width_space.word_ids == word_ids({0, 1}));
  assert((
      zero_width_space.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((
      zero_width_space.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(zero_width_space.ids, true) == "a b");

  const auto mixed_controls = tokenizer.encode("A\x0B" "B\x0C" "C", false);
  assert((mixed_controls.ids == std::vector<std::uint32_t>{5941, 272}));
  assert((mixed_controls.tokens == std::vector<std::string>{
                                  "\xE2\x96\x81" "ab", "\xE2\x96\x81" "c"}));
  assert((mixed_controls.offsets == std::vector<tokenizers_cpp::Offset>{
                                    {0, 3}, {4, 5}}));
  assert((mixed_controls.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(mixed_controls.word_ids == word_ids({0, 1}));
  assert((
      mixed_controls.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((
      mixed_controls.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(mixed_controls.ids, true) == "ab c");

  const auto combining_then_joiner =
      tokenizer.encode("e\xCC\x81" "\xE2\x80\x8D" "g", false);
  assert((
      combining_then_joiner.ids == std::vector<std::uint32_t>{13, 62, 489}));
  assert((combining_then_joiner.tokens == std::vector<std::string>{
                                             "\xE2\x96\x81",
                                             "e",
                                             "\xE2\x96\x81" "g"}));
  assert((combining_then_joiner.offsets == std::vector<tokenizers_cpp::Offset>{
                                                {0, 1}, {0, 1}, {6, 7}}));
  assert((
      combining_then_joiner.type_ids == std::vector<std::uint32_t>{0, 0, 0}));
  assert(combining_then_joiner.word_ids == word_ids({0, 0, 1}));
  assert((
      combining_then_joiner.special_tokens_mask ==
      std::vector<std::uint32_t>{0, 0, 0}));
  assert((
      combining_then_joiner.attention_mask ==
      std::vector<std::uint32_t>{1, 1, 1}));
  assert(tokenizer.decode(combining_then_joiner.ids, true) == "e g");

  const auto no_break_space = tokenizer.encode("A\xC2\xA0" "B", false);
  assert((no_break_space.ids == std::vector<std::uint32_t>{21, 334}));
  assert((no_break_space.tokens == std::vector<std::string>{
                                  "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((no_break_space.offsets == std::vector<tokenizers_cpp::Offset>{
                                    {0, 1}, {3, 4}}));
  assert((no_break_space.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(no_break_space.word_ids == word_ids({0, 1}));
  assert((
      no_break_space.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((
      no_break_space.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(no_break_space.ids, true) == "a b");

  const auto byte_order_mark = tokenizer.encode("A\xEF\xBB\xBF" "B", false);
  assert((byte_order_mark.ids == std::vector<std::uint32_t>{21, 334}));
  assert((byte_order_mark.tokens == std::vector<std::string>{
                                   "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((byte_order_mark.offsets == std::vector<tokenizers_cpp::Offset>{
                                     {0, 1}, {4, 5}}));
  assert((byte_order_mark.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(byte_order_mark.word_ids == word_ids({0, 1}));
  assert((
      byte_order_mark.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((
      byte_order_mark.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(byte_order_mark.ids, true) == "a b");

  const auto multiple_joiners =
      tokenizer.encode("A\xE2\x80\x8D\xE2\x80\x8C\xE2\x80\x8B" "B", false);
  assert((multiple_joiners.ids == std::vector<std::uint32_t>{21, 334}));
  assert((multiple_joiners.tokens == std::vector<std::string>{
                                    "\xE2\x96\x81" "a", "\xE2\x96\x81" "b"}));
  assert((multiple_joiners.offsets == std::vector<tokenizers_cpp::Offset>{
                                      {0, 1}, {10, 11}}));
  assert((multiple_joiners.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(multiple_joiners.word_ids == word_ids({0, 1}));
  assert((
      multiple_joiners.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((
      multiple_joiners.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(multiple_joiners.ids, true) == "a b");

  const auto ligature_expansion = tokenizer.encode("\xEF\xAC\x83", false);
  assert((ligature_expansion.ids == std::vector<std::uint32_t>{398, 1707}));
  assert((ligature_expansion.tokens == std::vector<std::string>{
                                       "\xE2\x96\x81" "f", "fi"}));
  assert((ligature_expansion.offsets == std::vector<tokenizers_cpp::Offset>{
                                         {0, 3}, {0, 3}}));
  assert((ligature_expansion.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(ligature_expansion.word_ids == word_ids({0, 0}));
  assert((
      ligature_expansion.special_tokens_mask ==
      std::vector<std::uint32_t>{0, 0}));
  assert((
      ligature_expansion.attention_mask == std::vector<std::uint32_t>{1, 1}));
  assert(tokenizer.decode(ligature_expansion.ids, true) == "ffi");

  const auto trademark_then_joiner =
      tokenizer.encode("\xE2\x84\xA2\xE2\x80\x8D" "g", false);
  assert((
      trademark_then_joiner.ids ==
      std::vector<std::uint32_t>{13, 38, 79, 489}));
  assert((trademark_then_joiner.tokens == std::vector<std::string>{
                                               "\xE2\x96\x81",
                                               "t",
                                               "m",
                                               "\xE2\x96\x81" "g"}));
  assert((trademark_then_joiner.offsets == std::vector<tokenizers_cpp::Offset>{
                                                 {0, 3}, {0, 3}, {0, 3}, {6, 7}}));
  assert((
      trademark_then_joiner.type_ids ==
      std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert(trademark_then_joiner.word_ids == word_ids({0, 0, 0, 1}));
  assert((
      trademark_then_joiner.special_tokens_mask ==
      std::vector<std::uint32_t>{0, 0, 0, 0}));
  assert((
      trademark_then_joiner.attention_mask ==
      std::vector<std::uint32_t>{1, 1, 1, 1}));
  assert(tokenizer.decode(trademark_then_joiner.ids, true) == "tm g");
}

void test_sentencepiece_nfkd_uses_icu_beyond_targeted_subset() {
  const auto tokenizer = load_unigram_sentencepiece_tokenizer(
      "tokenizers_cpp_sentencepiece_nfkd_icu.json",
      {
          {"type", "Sequence"},
          {"normalizers",
           json::array({
               {{"type", "NFKD"}},
           })},
      },
      unigram_vocab({
          "1",
          "\xE6\xA0\xAA" "\xE5\xBC\x8F" "\xE4\xBC\x9A" "\xE7\xA4\xBE",
      }));

  const auto output = tokenizer.encode(
      "\xE2\x91\xA0 \xE3\x8D\xBF",
      false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{
                               "1",
                               "\xE6\xA0\xAA" "\xE5\xBC\x8F" "\xE4\xBC\x9A" "\xE7\xA4\xBE"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{{0, 3}, {4, 7}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(output.word_ids == word_ids({0, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_sentencepiece_lowercase_uses_icu_beyond_targeted_subset() {
  const auto tokenizer = load_unigram_sentencepiece_tokenizer(
      "tokenizers_cpp_sentencepiece_lowercase_icu.json",
      {
          {"type", "Sequence"},
          {"normalizers",
           json::array({
               {{"type", "Lowercase"}},
           })},
      },
      unigram_vocab({
          "\xD5\xA1",
          "\xD5\xA2",
      }));

  const auto output = tokenizer.encode("\xD4\xB1 \xD4\xB2", false);
  assert((output.ids == std::vector<std::uint32_t>{1, 2}));
  assert((output.tokens == std::vector<std::string>{"\xD5\xA1", "\xD5\xA2"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{{0, 2}, {3, 5}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0}));
  assert(output.word_ids == word_ids({0, 1}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0, 0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1}));
}

void test_sentencepiece_lowercase_context_sensitive_sigma() {
  const auto tokenizer = load_unigram_sentencepiece_tokenizer(
      "tokenizers_cpp_sentencepiece_lowercase_sigma_icu.json",
      {
          {"type", "Sequence"},
          {"normalizers",
           json::array({
               {{"type", "Lowercase"}},
           })},
      },
      unigram_vocab({
          "\xCE\xBF\xCF\x82",
      }));

  const auto output = tokenizer.encode("\xCE\x9F\xCE\xA3", false);
  assert((output.ids == std::vector<std::uint32_t>{1}));
  assert((output.tokens == std::vector<std::string>{"\xCE\xBF\xCF\x82"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{{0, 4}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0}));
  assert(output.word_ids == word_ids({0}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{0}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1}));
}

void test_albert_pair_with_template_processing(
    const tokenizers_cpp::Tokenizer & tokenizer) {
  const auto output = tokenizer.encode_pair("Hello", "world", true);
  assert((output.ids == std::vector<std::uint32_t>{2, 10975, 3, 126, 3}));
  assert((output.tokens == std::vector<std::string>{
                            "[CLS]", "\xE2\x96\x81hello", "[SEP]", "\xE2\x96\x81world", "[SEP]"}));
  assert((output.offsets == std::vector<tokenizers_cpp::Offset>{
                                {0, 0}, {0, 5}, {0, 0}, {0, 5}, {0, 0}}));
  assert((output.type_ids == std::vector<std::uint32_t>{0, 0, 0, 1, 1}));
  assert(output.word_ids == word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt}));
  assert((output.special_tokens_mask == std::vector<std::uint32_t>{1, 0, 1, 0, 1}));
  assert((output.attention_mask == std::vector<std::uint32_t>{1, 1, 1, 1, 1}));
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "albert-base-v1-tokenizer.json");

  test_albert_single_without_specials(tokenizer);
  test_albert_single_with_template_processing(tokenizer);
  test_albert_nfkd_strip_accents(tokenizer);
  test_albert_precompiled_charsmap(tokenizer);
  test_sentencepiece_nfkd_uses_icu_beyond_targeted_subset();
  test_sentencepiece_lowercase_uses_icu_beyond_targeted_subset();
  test_sentencepiece_lowercase_context_sensitive_sigma();
  test_albert_pair_with_template_processing(tokenizer);
  return 0;
}
