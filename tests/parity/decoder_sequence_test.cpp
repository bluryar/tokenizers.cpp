#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

json base_tokenizer_json(const json & vocab, const json & decoder) {
  return {
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
}

tokenizers_cpp::Tokenizer load_tokenizer(
    const std::string & name,
    const json & vocab,
    const json & decoder) {
  const auto path = write_temp_tokenizer_json(
      name,
      base_tokenizer_json(vocab, decoder));
  auto tokenizer = tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
  return tokenizer;
}

void test_byte_fallback_fuse_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_byte_fallback.json",
      {
          {"[UNK]", 0},
          {"<0xE5>", 1},
          {"<0x8F>", 2},
          {"<0xAB>", 3},
          {"a", 4},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {{"type", "ByteFallback"}},
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2, 3, 4}, false) == "\xE5\x8F\xAB"
      "a");
  assert(tokenizer.decode({1, 2, 4}, false) == "\xEF\xBF\xBD\xEF\xBF\xBD"
      "a");
}

void test_ctc_metaspace_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_ctc_metaspace.json",
      {
          {"[UNK]", 0},
          {"\xE2\x96\x81", 1},
          {"H", 2},
          {"i", 3},
          {"y", 4},
          {"o", 5},
          {"u", 6},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {{"type", "CTC"}},
               {
                   {"type", "Metaspace"},
                   {"replacement", "\xE2\x96\x81"},
                   {"prepend_scheme", "always"},
                   {"split", true},
               },
           })},
      });

  assert(tokenizer.decode({1, 1, 2, 2, 3, 3, 1, 4, 5, 6}, false) == "Hi you");
}

void test_ctc_cleanup_fuse_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_ctc_cleanup.json",
      {
          {"[UNK]", 0},
          {"<pad>", 1},
          {"I", 2},
          {" 'm", 3},
          {"|", 4},
          {"fine", 5},
          {" !", 6},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "CTC"},
                   {"pad_token", "<pad>"},
                   {"word_delimiter_token", "|"},
                   {"cleanup", true},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 1}, false) ==
      "I'm fine!");
}

void test_ctc_custom_no_cleanup_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_ctc_no_cleanup.json",
      {
          {"[UNK]", 0},
          {"_", 1},
          {"A", 2},
          {"/", 3},
          {" .", 4},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "CTC"},
                   {"pad_token", "_"},
                   {"word_delimiter_token", "/"},
                   {"cleanup", false},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 1, 2, 2, 3, 3, 4, 4, 1}, false) == "A/ .");
}

void test_strip_replace_fuse_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_strip_replace.json",
      {
          {"[UNK]", 0},
          {"Hey", 1},
          {"_hello", 2},
          {"HHH", 3},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "Strip"},
                   {"content", "H"},
                   {"start", 1},
                   {"stop", 0},
               },
               {
                   {"type", "Replace"},
                   {"pattern", {{"String", "_"}}},
                   {"content", " "},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2, 3}, false) == "ey helloHH");
}

void test_replace_regex_sequence() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_sequence_replace_regex.json",
      {
          {"[UNK]", 0},
          {"This     ", 1},
          {"is", 2},
          {"   fine", 3},
      },
      {
          {"type", "Sequence"},
          {"decoders",
           json::array({
               {
                   {"type", "Replace"},
                   {"pattern", {{"Regex", "\\s+"}}},
                   {"content", " "},
               },
               {{"type", "Fuse"}},
           })},
      });

  assert(tokenizer.decode({1, 2, 3}, false) == "This is fine");
}

void test_bpe_decoder() {
  const auto tokenizer = load_tokenizer(
      "tokenizers_cpp_decoder_bpe.json",
      {
          {"[UNK]", 0},
          {"low</w>", 1},
          {"er</w>", 2},
      },
      {
          {"type", "BPEDecoder"},
          {"suffix", "</w>"},
      });

  assert(tokenizer.decode({1, 2}, false) == "low er");
}

}  // namespace

int main() {
  test_byte_fallback_fuse_sequence();
  test_ctc_metaspace_sequence();
  test_ctc_cleanup_fuse_sequence();
  test_ctc_custom_no_cleanup_sequence();
  test_strip_replace_fuse_sequence();
  test_replace_regex_sequence();
  test_bpe_decoder();
  return 0;
}
