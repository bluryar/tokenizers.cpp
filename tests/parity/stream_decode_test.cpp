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

json added_token(const std::string & content, bool normalized, bool special = false) {
  return {
      {"id", 0},
      {"content", content},
      {"single_word", false},
      {"lstrip", false},
      {"rstrip", false},
      {"normalized", normalized},
      {"special", special},
  };
}

tokenizers_cpp::Tokenizer load_llama_tokenizer(const std::filesystem::path & data_dir) {
  return tokenizers_cpp::Tokenizer::from_file(data_dir / "llama-3-tokenizer.json");
}

std::string stream_llama_split_pattern() {
  return R"regex((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)regex";
}

std::vector<tokenizers_cpp::Offset> offsets(
    std::initializer_list<tokenizers_cpp::Offset> values) {
  return values;
}

std::vector<std::optional<std::uint32_t>> word_ids(
    std::initializer_list<std::optional<std::uint32_t>> values) {
  return values;
}

void assert_step(
    tokenizers_cpp::DecodeStream & stream,
    std::uint32_t id,
    const std::string & expected) {
  const auto output = stream.step(id);
  assert(output.has_value());
  assert(output.value() == expected);
}

void assert_pending(tokenizers_cpp::DecodeStream & stream, std::uint32_t id) {
  const auto output = stream.step(id);
  assert(!output.has_value());
}

void test_decoding_with_added_bpe(const std::filesystem::path & data_dir) {
  auto tokenizer = load_llama_tokenizer(data_dir);
  tokenizer.with_byte_level_normalizer();
  tokenizer.with_split_pre_tokenizer(stream_llama_split_pattern());

  tokenizers_cpp::AddedToken ma;
  ma.content = "\xE5\x97\x8E";
  ma.normalized = false;
  assert(tokenizer.add_tokens({ma}) == 1);
  assert(tokenizer.token_to_id("\xE5\x97\x8E").value() == 128256U);

  const auto ma_encoded =
      tokenizer.encode("Hey! how is this token: \xE5\x97\x8E", false);
  assert((ma_encoded.ids == std::vector<std::uint32_t>{
                                19182,
                                0,
                                1268,
                                602,
                                82,
                                62428,
                                82,
                                4037,
                                25,
                                220,
                                128256}));
  assert((ma_encoded.tokens == std::vector<std::string>{
                                   "Hey",
                                   "!",
                                   "\xC4\xA0how",
                                   "\xC4\xA0i",
                                   "s",
                                   "\xC4\xA0thi",
                                   "s",
                                   "\xC4\xA0token",
                                   ":",
                                   "\xC4\xA0",
                                   "\xE5\x97\x8E"}));
  assert((ma_encoded.offsets == offsets({
                                    {0, 3},
                                    {3, 4},
                                    {4, 8},
                                    {8, 10},
                                    {10, 11},
                                    {11, 15},
                                    {15, 16},
                                    {16, 22},
                                    {22, 23},
                                    {23, 24},
                                    {24, 27},
                                })));
  assert((ma_encoded.type_ids == std::vector<std::uint32_t>{
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert(ma_encoded.word_ids == word_ids({0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 5}));
  assert((ma_encoded.special_tokens_mask == std::vector<std::uint32_t>{
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert((ma_encoded.attention_mask == std::vector<std::uint32_t>{
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  assert(
      tokenizer.decode(
          {19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128256},
          false) == "Hey! how is this token: \xE5\x97\x8E");

  tokenizers_cpp::AddedToken de;
  de.content = "\xD0\xB4";
  de.normalized = true;
  assert(tokenizer.add_tokens({de}) == 1);
  assert(tokenizer.token_to_id("\xD0\xB4").value() == 128257U);

  const auto de_encoded =
      tokenizer.encode("Hey! how is this token: \xD0\xB4", false);
  assert((de_encoded.ids == std::vector<std::uint32_t>{
                                19182,
                                0,
                                1268,
                                602,
                                82,
                                62428,
                                82,
                                4037,
                                25,
                                220,
                                128257}));
  assert((de_encoded.tokens == std::vector<std::string>{
                                   "Hey",
                                   "!",
                                   "\xC4\xA0how",
                                   "\xC4\xA0i",
                                   "s",
                                   "\xC4\xA0thi",
                                   "s",
                                   "\xC4\xA0token",
                                   ":",
                                   "\xC4\xA0",
                                   "\xC3\x90\xC2\xB4"}));
  assert((de_encoded.offsets == offsets({
                                    {0, 3},
                                    {3, 4},
                                    {4, 8},
                                    {8, 10},
                                    {10, 11},
                                    {11, 15},
                                    {15, 16},
                                    {16, 22},
                                    {22, 23},
                                    {23, 24},
                                    {24, 26},
                                })));
  assert((de_encoded.type_ids == std::vector<std::uint32_t>{
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert(de_encoded.word_ids == word_ids({0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 5}));
  assert((de_encoded.special_tokens_mask == std::vector<std::uint32_t>{
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  assert((de_encoded.attention_mask == std::vector<std::uint32_t>{
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
  assert(
      tokenizer.decode(
          {19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128257},
          false) == "Hey! how is this token: \xD0\xB4");
}

void test_decode_stream_step_no_panic(const std::filesystem::path & data_dir) {
  const auto tokenizer = load_llama_tokenizer(data_dir);

  auto ascii_stream = tokenizer.decode_stream(false);
  assert(!ascii_stream.has_pending());
  assert_step(ascii_stream, 32, "A");
  assert(!ascii_stream.has_pending());
  assert_step(ascii_stream, 426, " B");
  assert_step(ascii_stream, 356, " C");
  assert_step(ascii_stream, 423, " D");
  assert_step(ascii_stream, 469, " E");
  assert_step(ascii_stream, 435, " F");
  assert_step(ascii_stream, 480, " G");
  assert_step(ascii_stream, 473, " H");
  assert_step(ascii_stream, 358, " I");
  assert_step(ascii_stream, 622, " J");

  auto korean_stream = tokenizer.decode_stream(false);
  assert(!korean_stream.has_pending());
  assert_pending(korean_stream, 80690);
  assert(korean_stream.has_pending());
  assert_step(korean_stream, 98, "\xEC\x82\xA5");
  assert(!korean_stream.has_pending());
  assert_pending(korean_stream, 167);
  assert(korean_stream.has_pending());
  assert_pending(korean_stream, 121);
  assert(korean_stream.has_pending());
  assert_step(korean_stream, 243, "\xEB\xBD\x95");
  assert(!korean_stream.has_pending());
  assert_pending(korean_stream, 102457);
  assert(korean_stream.has_pending());
  assert_step(korean_stream, 113, "\xEB\xB9\xB5");
  assert(!korean_stream.has_pending());
}

void test_documentation_streaming_roberta(const std::filesystem::path & data_dir) {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file(data_dir / "roberta.json");
  auto stream = tokenizer.decode_stream(false);
  assert_step(stream, 713, "This");
  assert_step(stream, 16, " is");
  assert_step(stream, 41, " an");
  assert_step(stream, 1246, " example");
}

void test_documentation_streaming_albert(const std::filesystem::path & data_dir) {
  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(
      data_dir / "albert-base-v1-tokenizer.json");

  const auto encoded = tokenizer.encode("This is an example", false);
  assert((encoded.ids == std::vector<std::uint32_t>{48, 25, 40, 823}));

  auto starts_at_is = tokenizer.decode_stream(false);
  assert_step(starts_at_is, 25, "is");

  auto stream = tokenizer.decode_stream(false);
  assert_step(stream, 48, "this");
  assert_step(stream, 25, " is");
  assert_step(stream, 40, " an");
  assert_step(stream, 823, " example");
}

tokenizers_cpp::Tokenizer load_byte_fallback_stream_tokenizer() {
  const json value = {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer",
       {
           {"type", "Sequence"},
           {"normalizers",
            json::array({
                {
                    {"type", "Strip"},
                    {"strip_left", true},
                    {"strip_right", true},
                },
                {
                    {"type", "NFC"},
                },
            })},
       }},
      {"pre_tokenizer",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", false},
           {"trim_offsets", true},
           {"use_regex", true},
       }},
      {"post_processor",
       {
           {"type", "ByteLevel"},
           {"add_prefix_space", false},
           {"trim_offsets", true},
           {"use_regex", true},
       }},
      {"decoder",
       {
           {"type", "ByteFallback"},
       }},
      {"model",
       {
           {"type", "BPE"},
           {"dropout", nullptr},
           {"unk_token", nullptr},
           {"continuing_subword_prefix", nullptr},
           {"end_of_word_suffix", nullptr},
           {"fuse_unk", false},
           {"byte_fallback", true},
           {"ignore_merges", false},
           {"vocab",
            {
                {"<0x20>", 0},
                {"<0xC3>", 1},
                {"<0xA9>", 2},
                {" This", 3},
            }},
           {"merges", json::array()},
       }},
  };

  const auto path = write_temp_tokenizer_json(
      "tokenizers_cpp_documentation_streaming_byte_fallback.json",
      value);
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

tokenizers_cpp::Tokenizer load_wordlevel_stream_tokenizer(
    const std::string & name,
    const json & vocab,
    const json & decoder) {
  const json value = {
      {"version", "1.0"},
      {"truncation", nullptr},
      {"padding", nullptr},
      {"added_tokens", json::array()},
      {"normalizer", nullptr},
      {"pre_tokenizer", nullptr},
      {"post_processor", nullptr},
      {"decoder", decoder},
      {"model",
       {
           {"type", "WordLevel"},
           {"unk_token", "[UNK]"},
           {"vocab", vocab},
       }},
  };

  const auto path = write_temp_tokenizer_json(name, value);
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_documentation_streaming_byte_fallback_none() {
  const auto tokenizer = load_byte_fallback_stream_tokenizer();
  auto stream = tokenizer.decode_stream(false);
  assert(!stream.has_pending());
  assert_step(stream, 0, " ");
  assert(!stream.has_pending());
  assert_pending(stream, 1);
  assert(stream.has_pending());
  assert_step(stream, 2, "\xC3\xA9");
  assert(!stream.has_pending());
  assert_pending(stream, 2);
  assert(stream.has_pending());
  assert_step(stream, 3, "\xEF\xBF\xBD This");
  assert(!stream.has_pending());
}

void test_stream_sequence_byte_fallback_fuse() {
  const auto tokenizer = load_wordlevel_stream_tokenizer(
      "tokenizers_cpp_stream_sequence_byte_fallback_fuse.json",
      {
          {"[UNK]", 0},
          {"<0xC3>", 1},
          {"<0xA9>", 2},
          {" This", 3},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {{"type", "ByteFallback"}},
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2, 3}, false) == "\xC3\xA9 This");

  auto stream = tokenizer.decode_stream(false);
  assert(!stream.has_pending());
  assert_pending(stream, 1);
  assert(stream.has_pending());
  assert_step(stream, 2, "\xC3\xA9");
  assert(!stream.has_pending());
  assert_step(stream, 3, " This");

  auto orphan = tokenizer.decode_stream(false);
  assert_pending(orphan, 2);
  assert(orphan.has_pending());
  assert_step(orphan, 3, "\xEF\xBF\xBD This");
  assert(!orphan.has_pending());
}

void test_stream_sequence_byte_fallback_has_no_finalize() {
  const auto tokenizer = load_wordlevel_stream_tokenizer(
      "tokenizers_cpp_stream_sequence_byte_fallback_no_finalize.json",
      {
          {"[UNK]", 0},
          {"<0xC3>", 1},
          {" This", 2},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {{"type", "ByteFallback"}},
               {{"type", "Fuse"}},
           })},
      });

  auto stream = tokenizer.decode_stream(false);
  assert_pending(stream, 1);
  assert(stream.has_pending());

  auto reset = tokenizer.decode_stream(false);
  assert(!reset.has_pending());
  assert_step(reset, 2, " This");
  assert(!reset.has_pending());
}

void test_stream_sequence_metaspace_fuse() {
  const auto tokenizer = load_wordlevel_stream_tokenizer(
      "tokenizers_cpp_stream_sequence_metaspace_fuse.json",
      {
          {"[UNK]", 0},
          {"\xE2\x96\x81Hello", 1},
          {"\xE2\x96\x81world", 2},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "Metaspace"},
                   {"replacement", "\xE2\x96\x81"},
                   {"prepend_scheme", "always"},
                   {"split", true},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2}, false) == "Hello world");

  auto stream = tokenizer.decode_stream(false);
  assert_step(stream, 1, "Hello");
  assert(!stream.has_pending());
  assert_step(stream, 2, " world");
  assert(!stream.has_pending());
}

void test_stream_sequence_byte_level_fuse() {
  const auto tokenizer = load_wordlevel_stream_tokenizer(
      "tokenizers_cpp_stream_sequence_byte_level_fuse.json",
      {
          {"[UNK]", 0},
          {"Hello", 1},
          {"\xC4\xA0world", 2},
          {"\xC3\x83", 3},
          {"\xC2\xA9", 4},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "ByteLevel"},
                   {"add_prefix_space", false},
                   {"trim_offsets", true},
                   {"use_regex", true},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2}, false) == "Hello world");
  assert(tokenizer.decode({3, 4}, false) == "\xC3\xA9");

  auto words = tokenizer.decode_stream(false);
  assert_step(words, 1, "Hello");
  assert(!words.has_pending());
  assert_step(words, 2, " world");
  assert(!words.has_pending());

  auto unicode = tokenizer.decode_stream(false);
  assert_pending(unicode, 3);
  assert(unicode.has_pending());
  assert_step(unicode, 4, "\xC3\xA9");
  assert(!unicode.has_pending());
}

}  // namespace

int main() {
  const auto data_dir = std::filesystem::path(TOKENIZERS_CPP_HF_TEST_DATA_DIR);
  test_decoding_with_added_bpe(data_dir);
  test_decode_stream_step_no_panic(data_dir);
  test_documentation_streaming_roberta(data_dir);
  test_documentation_streaming_albert(data_dir);
  test_documentation_streaming_byte_fallback_none();
  test_stream_sequence_byte_fallback_fuse();
  test_stream_sequence_byte_fallback_has_no_finalize();
  test_stream_sequence_metaspace_fuse();
  test_stream_sequence_byte_level_fuse();
  return 0;
}
