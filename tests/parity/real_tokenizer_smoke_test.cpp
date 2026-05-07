#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
  assert(input && "failed to open tokenizer JSON fixture");
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

std::vector<tokenizers_cpp::Offset> offsets(
    std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

void assert_encoding(
    const tokenizers_cpp::Encoding & output,
    const std::vector<std::uint32_t> & ids,
    const std::vector<std::string> & tokens,
    const std::vector<tokenizers_cpp::Offset> & output_offsets,
    const std::vector<std::uint32_t> & type_ids,
    const std::vector<std::optional<std::uint32_t>> & output_word_ids,
    const std::vector<std::uint32_t> & special_tokens_mask,
    const std::vector<std::uint32_t> & attention_mask) {
  assert(output.ids == ids);
  assert(output.tokens == tokens);
  assert(output.offsets == output_offsets);
  assert(output.type_ids == type_ids);
  assert(output.word_ids == output_word_ids);
  assert(output.special_tokens_mask == special_tokens_mask);
  assert(output.attention_mask == attention_mask);
}

tokenizers_cpp::Tokenizer load_gpt_style_sequence_tokenizer(
    const std::filesystem::path & data_dir) {
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

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_real_smoke_gpt_sequence.json",
      tokenizer_json);
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_albert_truncation_padding_tokenizer(
    const std::filesystem::path & data_dir) {
  auto tokenizer_json = read_json(data_dir / "albert-base-v1-tokenizer.json");
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
      {"pad_id", 0},
      {"pad_type_id", 0},
      {"pad_token", "<pad>"},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_real_smoke_albert_truncation_padding.json",
      tokenizer_json);
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void assert_gpt_style_the_a_output(const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {0, 90, 81},
      {"<s>", "\xC4\xA0the", "\xC4\xA0" "a"},
      offsets({{0, 0}, {0, 3}, {4, 5}}),
      {0, 0, 0},
      word_ids({std::nullopt, 0, 1}),
      {1, 0, 0},
      {1, 1, 1});
  assert(output.overflowing.empty());
}

void assert_gpt_style_the_output(const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {0, 90},
      {"<s>", "\xC4\xA0the"},
      offsets({{0, 0}, {0, 3}}),
      {0, 0},
      word_ids({std::nullopt, 0}),
      {1, 0},
      {1, 1});
  assert(output.overflowing.empty());
}

void assert_gpt_style_pair_output(
    const tokenizers_cpp::Encoding & output,
    std::uint32_t first_id,
    const std::string & first_token,
    std::uint32_t second_id,
    const std::string & second_token,
    tokenizers_cpp::Offset first_offset,
    tokenizers_cpp::Offset second_offset) {
  assert_encoding(
      output,
      {0, first_id, 0, second_id},
      {"<s>", first_token, "<s>", second_token},
      offsets({{0, 0}, first_offset, {0, 0}, second_offset}),
      {0, 0, 1, 1},
      word_ids({std::nullopt, 0, std::nullopt, 0}),
      {1, 0, 1, 0},
      {1, 1, 1, 1});
  assert(output.overflowing.empty());
}

void test_gpt_style_bytelevel_bpe_sequence(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_gpt_style_sequence_tokenizer(data_dir);
  const auto output = tokenizer.encode("the a", true);
  assert_gpt_style_the_a_output(output);
  assert(tokenizer.decode(output.ids, true) == " the a");
}

void test_gpt_style_bytelevel_bpe_sequence_batch(
    const std::filesystem::path & data_dir) {
  const auto tokenizer = load_gpt_style_sequence_tokenizer(data_dir);
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"the a", "the"},
      true);

  assert(outputs.size() == 2);
  assert_gpt_style_the_a_output(outputs[0]);
  assert_gpt_style_the_output(outputs[1]);
  assert((tokenizer.decode_batch({outputs[0].ids, outputs[1].ids}, true) ==
          std::vector<std::string>{" the a", " the"}));
  assert((tokenizer.decode_batch({outputs[0].ids, outputs[1].ids}, false) ==
          std::vector<std::string>{"<s> the a", "<s> the"}));
}

void test_gpt_style_bytelevel_bpe_sequence_pair_batch(
    const std::filesystem::path & data_dir) {
  const auto tokenizer = load_gpt_style_sequence_tokenizer(data_dir);
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"the", "a"},
          {"a", "the"},
      },
      true);

  assert(outputs.size() == 2);
  assert_gpt_style_pair_output(
      outputs[0],
      90,
      "\xC4\xA0the",
      81,
      "\xC4\xA0" "a",
      {0, 3},
      {0, 1});
  assert_gpt_style_pair_output(
      outputs[1],
      81,
      "\xC4\xA0" "a",
      90,
      "\xC4\xA0the",
      {0, 1},
      {0, 3});
}

void test_roberta_bytelevel_bpe(const std::filesystem::path & data_dir) {
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");
  const auto output = tokenizer.encode("Hello world", true);
  assert_encoding(
      output,
      {0, 31414, 232, 2},
      {"<s>", "Hello", "\xC4\xA0world", "</s>"},
      offsets({{0, 0}, {0, 5}, {6, 11}, {0, 0}}),
      {0, 0, 0, 0},
      word_ids({std::nullopt, 0, 1, std::nullopt}),
      {1, 0, 0, 1},
      {1, 1, 1, 1});
  assert(tokenizer.decode(output.ids, true) == "Hello world");
}

void assert_roberta_pair_output(
    const tokenizers_cpp::Encoding & output,
    std::uint32_t second_id,
    const std::string & second_token,
    tokenizers_cpp::Offset second_offset) {
  assert_encoding(
      output,
      {0, 31414, 2, 2, second_id, 2},
      {"<s>", "Hello", "</s>", "</s>", second_token, "</s>"},
      offsets({{0, 0}, {0, 5}, {0, 0}, {0, 0}, second_offset, {0, 0}}),
      {0, 0, 0, 0, 0, 0},
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, 0, std::nullopt}),
      {1, 0, 1, 1, 0, 1},
      {1, 1, 1, 1, 1, 1});
  assert(output.overflowing.empty());
}

void test_roberta_bytelevel_bpe_pair_batch(const std::filesystem::path & data_dir) {
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"Hello", "world"},
          {"Hello", "test"},
      },
      true);

  assert(outputs.size() == 2);
  assert_roberta_pair_output(outputs[0], 8331, "world", {0, 5});
  assert_roberta_pair_output(outputs[1], 21959, "test", {0, 4});
}

void test_bert_wordpiece(const std::filesystem::path & data_dir) {
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(data_dir / "bert-wiki.json");
  const auto output = tokenizer.encode(
      "Welcome to the \xF0\x9F\xA4\x97 Tokenizers library.",
      true);
  assert_encoding(
      output,
      {1, 18263, 7128, 7108, 0, 22453, 27107, 12800, 4068, 11046, 18, 2},
      {"[CLS]",
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
       "[SEP]"},
      offsets({
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
      }),
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      word_ids({std::nullopt, 0, 1, 2, 3, 4, 4, 4, 4, 5, 6, std::nullopt}),
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  assert(tokenizer.decode(output.ids, true) ==
      "welcome to the tok ##eni ##zer ##s library .");
  tokenizer.with_wordpiece_decoder();
  assert(tokenizer.decode(output.ids, true) == "welcome to the tokenizers library.");
}

void assert_bert_pair_output(
    const tokenizers_cpp::Encoding & output,
    std::uint32_t second_id,
    const std::string & second_token,
    tokenizers_cpp::Offset second_offset) {
  assert_encoding(
      output,
      {1, 27462, 2, second_id, 2},
      {"[CLS]", "hello", "[SEP]", second_token, "[SEP]"},
      offsets({{0, 0}, {0, 5}, {0, 0}, second_offset, {0, 0}}),
      {0, 0, 0, 1, 1},
      word_ids({std::nullopt, 0, std::nullopt, 0, std::nullopt}),
      {1, 0, 1, 0, 1},
      {1, 1, 1, 1, 1});
  assert(output.overflowing.empty());
}

void test_bert_wordpiece_pair_batch(const std::filesystem::path & data_dir) {
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(data_dir / "bert-wiki.json");
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"Hello", "world"},
          {"Hello", "test"},
      },
      true);

  assert(outputs.size() == 2);
  assert_bert_pair_output(outputs[0], 7601, "world", {0, 5});
  assert_bert_pair_output(outputs[1], 8396, "test", {0, 4});

  tokenizer.with_wordpiece_decoder();
  assert((tokenizer.decode_batch({outputs[0].ids, outputs[1].ids}, true) ==
          std::vector<std::string>{"hello world", "hello test"}));
}

void test_albert_sentencepiece_unigram(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "albert-base-v1-tokenizer.json");
  const auto output = tokenizer.encode("Hello world", true);
  assert_encoding(
      output,
      {2, 10975, 126, 3},
      {"[CLS]", "\xE2\x96\x81hello", "\xE2\x96\x81world", "[SEP]"},
      offsets({{0, 0}, {0, 5}, {6, 11}, {0, 0}}),
      {0, 0, 0, 0},
      word_ids({std::nullopt, 0, 1, std::nullopt}),
      {1, 0, 0, 1},
      {1, 1, 1, 1});
  assert(tokenizer.decode(output.ids, true) == "hello world");
}

void assert_albert_truncated_padded_output(
    const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {2, 10975, 126, 3, 0, 0},
      {"[CLS]",
       "\xE2\x96\x81hello",
       "\xE2\x96\x81world",
       "[SEP]",
       "<pad>",
       "<pad>"},
      offsets({{0, 0}, {0, 5}, {6, 11}, {0, 0}, {0, 0}, {0, 0}}),
      {0, 0, 0, 0, 0, 0},
      word_ids({std::nullopt, 0, 1, std::nullopt, std::nullopt, std::nullopt}),
      {1, 0, 0, 1, 1, 1},
      {1, 1, 1, 1, 0, 0});

  assert(output.overflowing.size() == 1);
  const auto & overflow = output.overflowing.front();
  assert_encoding(
      overflow,
      {2, 1289, 3, 0, 0, 0},
      {"[CLS]",
       "\xE2\x96\x81test",
       "[SEP]",
       "<pad>",
       "<pad>",
       "<pad>"},
      offsets({{0, 0}, {12, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}),
      {0, 0, 0, 0, 0, 0},
      word_ids({std::nullopt, 2, std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
      {1, 0, 1, 1, 1, 1},
      {1, 1, 1, 0, 0, 0});
  assert(overflow.overflowing.empty());
}

void assert_albert_short_padded_output(
    const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {2, 10975, 3, 0, 0, 0},
      {"[CLS]",
       "\xE2\x96\x81hello",
       "[SEP]",
       "<pad>",
       "<pad>",
       "<pad>"},
      offsets({{0, 0}, {0, 5}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}),
      {0, 0, 0, 0, 0, 0},
      word_ids({std::nullopt, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
      {1, 0, 1, 1, 1, 1},
      {1, 1, 1, 0, 0, 0});
  assert(output.overflowing.empty());
}

void test_albert_sentencepiece_truncation_padding_overflow(
    const std::filesystem::path & data_dir) {
  const auto tokenizer = load_albert_truncation_padding_tokenizer(data_dir);
  const auto output = tokenizer.encode("Hello world test", true);
  assert_albert_truncated_padded_output(output);
  assert(tokenizer.decode(output.ids, true) == "hello world");
  assert(tokenizer.decode(output.ids, false) == "[CLS] hello world[SEP]<pad><pad>");
  const auto & overflow = output.overflowing.front();
  assert(tokenizer.decode(overflow.ids, true) == "test");
  assert(tokenizer.decode(overflow.ids, false) == "[CLS] test[SEP]<pad><pad><pad>");
}

void test_albert_sentencepiece_batch_truncation_padding_decode(
    const std::filesystem::path & data_dir) {
  const auto tokenizer = load_albert_truncation_padding_tokenizer(data_dir);
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"Hello world test", "Hello"},
      true);

  assert(outputs.size() == 2);
  assert_albert_truncated_padded_output(outputs[0]);
  assert_albert_short_padded_output(outputs[1]);

  const std::vector<std::vector<std::uint32_t>> batch_ids{
      outputs[0].ids,
      outputs[1].ids,
      outputs[0].overflowing.front().ids,
  };
  assert((tokenizer.decode_batch(batch_ids, true) ==
          std::vector<std::string>{"hello world", "hello", "test"}));
  assert((tokenizer.decode_batch(batch_ids, false) ==
          std::vector<std::string>{
              "[CLS] hello world[SEP]<pad><pad>",
              "[CLS] hello[SEP]<pad><pad><pad>",
              "[CLS] test[SEP]<pad><pad><pad>",
          }));
}

void assert_llama_hello_world_output(const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {128000, 9906, 11, 1917, 0},
      {"<|begin_of_text|>", "Hello", ",", "\xC4\xA0world", "!"},
      offsets({{0, 0}, {0, 5}, {5, 6}, {6, 12}, {12, 13}}),
      {0, 0, 0, 0, 0},
      word_ids({std::nullopt, 0, 1, 2, 3}),
      {1, 0, 0, 0, 0},
      {1, 1, 1, 1, 1});
  assert(output.overflowing.empty());
}

void assert_llama_hello_output(const tokenizers_cpp::Encoding & output) {
  assert_encoding(
      output,
      {128000, 9906},
      {"<|begin_of_text|>", "Hello"},
      offsets({{0, 0}, {0, 5}}),
      {0, 0},
      word_ids({std::nullopt, 0}),
      {1, 0},
      {1, 1});
  assert(output.overflowing.empty());
}

void assert_llama_pair_output(
    const tokenizers_cpp::Encoding & output,
    std::uint32_t second_id,
    const std::string & second_token,
    tokenizers_cpp::Offset second_offset) {
  assert_encoding(
      output,
      {128000, 9906, 128000, second_id},
      {"<|begin_of_text|>", "Hello", "<|begin_of_text|>", second_token},
      offsets({{0, 0}, {0, 5}, {0, 0}, second_offset}),
      {0, 0, 1, 1},
      word_ids({std::nullopt, 0, std::nullopt, 0}),
      {1, 0, 1, 0},
      {1, 1, 1, 1});
  assert(output.overflowing.empty());
}

void test_llama_split_bytelevel_bpe(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");
  const auto output = tokenizer.encode("Hello, world!", true);
  assert_llama_hello_world_output(output);
  assert(tokenizer.decode(output.ids, true) == "Hello, world!");
}

void test_llama_split_bytelevel_bpe_batch(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");
  const auto outputs = tokenizer.encode_batch(
      std::vector<std::string>{"Hello, world!", "Hello"},
      true);

  assert(outputs.size() == 2);
  assert_llama_hello_world_output(outputs[0]);
  assert_llama_hello_output(outputs[1]);
  assert((tokenizer.decode_batch({outputs[0].ids, outputs[1].ids}, true) ==
          std::vector<std::string>{"Hello, world!", "Hello"}));
}

void test_llama_split_bytelevel_bpe_pair_batch(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");
  const auto outputs = tokenizer.encode_batch_pairs(
      std::vector<std::pair<std::string, std::string>>{
          {"Hello", "world"},
          {"Hello", "test"},
      },
      true);

  assert(outputs.size() == 2);
  assert_llama_pair_output(outputs[0], 14957, "world", {0, 5});
  assert_llama_pair_output(outputs[1], 1985, "test", {0, 4});
  assert((tokenizer.decode_batch({outputs[0].ids, outputs[1].ids}, true) ==
          std::vector<std::string>{"Helloworld", "Hellotest"}));
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_gpt_style_bytelevel_bpe_sequence(data_dir);
  test_gpt_style_bytelevel_bpe_sequence_batch(data_dir);
  test_gpt_style_bytelevel_bpe_sequence_pair_batch(data_dir);
  test_roberta_bytelevel_bpe(data_dir);
  test_roberta_bytelevel_bpe_pair_batch(data_dir);
  test_bert_wordpiece(data_dir);
  test_bert_wordpiece_pair_batch(data_dir);
  test_albert_sentencepiece_unigram(data_dir);
  test_albert_sentencepiece_truncation_padding_overflow(data_dir);
  test_albert_sentencepiece_batch_truncation_padding_decode(data_dir);
  test_llama_split_bytelevel_bpe(data_dir);
  test_llama_split_bytelevel_bpe_batch(data_dir);
  test_llama_split_bytelevel_bpe_pair_batch(data_dir);
  return 0;
}
