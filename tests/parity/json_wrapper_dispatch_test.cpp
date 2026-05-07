#include "tokenizers_cpp/tokenizer.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::filesystem::path write_tokenizer_json(const std::string & name, const std::string & json) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path);
  out << json;
  return path;
}

void expect_loads(const std::string & name, const std::string & json) {
  const auto path = write_tokenizer_json(name, json);
  (void)tokenizers_cpp::Tokenizer::from_file(path);
  std::filesystem::remove(path);
}

void expect_load_fails(
    const std::string & name,
    const std::string & json,
    const std::string & message_part) {
  const auto path = write_tokenizer_json(name, json);
  try {
    (void)tokenizers_cpp::Tokenizer::from_file(path);
    assert(false && "Tokenizer::from_file should have rejected this JSON");
  } catch (const std::runtime_error & err) {
    assert(std::string(err.what()).find(message_part) != std::string::npos);
  }
  std::filesystem::remove(path);
}

std::string tokenizer_with_model(const std::string & model_json) {
  return R"json({
    "version": "1.0",
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": null,
    "model": )json" + model_json + "\n}";
}

std::string tokenizer_with_slot(const std::string & slot, const std::string & wrapper_json) {
  std::string json = R"json({
    "version": "1.0",
    "model": {
      "type": "WordLevel",
      "unk_token": "[UNK]",
      "vocab": {
        "[UNK]": 0,
        "hello": 1,
        "world": 2
      }
    })json";

  const std::vector<std::string> slots = {
      "normalizer",
      "pre_tokenizer",
      "post_processor",
      "decoder",
  };
  for (const auto & current_slot : slots) {
    json += ",\n    \"" + current_slot + "\": ";
    json += current_slot == slot ? wrapper_json : "null";
  }
  json += "\n}";
  return json;
}

}  // namespace

int main() {
  const std::vector<std::string> model_jsons = {
      R"json({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]})json",
      R"json({"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0,"hello":1}})json",
      R"json({"type":"WordLevel","unk_token":"[UNK]","vocab":{"[UNK]":0,"hello":1}})json",
      R"json({"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["hello",-1.0]]})json",
  };
  for (std::size_t i = 0; i < model_jsons.size(); ++i) {
    expect_loads(
        "tokenizers_cpp_model_dispatch_" + std::to_string(i) + ".json",
        tokenizer_with_model(model_jsons[i]));
  }

  expect_loads(
      "tokenizers_cpp_normalizer_nfc_dispatch.json",
      tokenizer_with_slot("normalizer", R"json({"type":"NFC"})json"));
  expect_loads(
      "tokenizers_cpp_normalizer_bert_dispatch.json",
      tokenizer_with_slot(
          "normalizer",
          R"json({"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true})json"));
  expect_loads(
      "tokenizers_cpp_normalizer_sequence_dispatch.json",
      tokenizer_with_slot(
          "normalizer",
          R"json({"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Lowercase"}]})json"));
  expect_loads(
      "tokenizers_cpp_tokenizer_wordpiece_nfc_dispatch.json",
      R"json({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {"type":"NFC"},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"hello":1}}
      })json");

  expect_loads(
      "tokenizers_cpp_pretok_bert_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"BertPreTokenizer"})json"));
  expect_loads(
      "tokenizers_cpp_pretok_delimiter_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"CharDelimiterSplit","delimiter":" "})json"));
  expect_loads(
      "tokenizers_cpp_pretok_whitespace_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"Whitespace"})json"));
  expect_loads(
      "tokenizers_cpp_pretok_split_string_dispatch.json",
      tokenizer_with_slot(
          "pre_tokenizer",
          R"json({"type":"Split","pattern":{"String":"[SEP]"},"behavior":"Isolated","invert":false})json"));
  expect_loads(
      "tokenizers_cpp_pretok_split_regex_dispatch.json",
      tokenizer_with_slot(
          "pre_tokenizer",
          R"json({"type":"Split","pattern":{"Regex":"[SEP]"},"behavior":"Isolated","invert":false})json"));
  expect_loads(
      "tokenizers_cpp_pretok_sequence_dispatch.json",
      tokenizer_with_slot(
          "pre_tokenizer",
          R"json({"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"_","prepend_scheme":"always","split":true}]})json"));

  expect_loads(
      "tokenizers_cpp_processor_bert_dispatch.json",
      tokenizer_with_slot(
          "post_processor",
          R"json({"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]})json"));
  expect_loads(
      "tokenizers_cpp_processor_bert_serialization_dispatch.json",
      tokenizer_with_slot(
          "post_processor",
          R"json({"type":"BertProcessing","sep":["SEP",0],"cls":["CLS",0]})json"));
  expect_loads(
      "tokenizers_cpp_processor_sequence_dispatch.json",
      tokenizer_with_slot(
          "post_processor",
          R"json({"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]})json"));

  expect_loads(
      "tokenizers_cpp_decoder_bytelevel_dispatch.json",
      tokenizer_with_slot(
          "decoder",
          R"json({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true})json"));
  expect_loads(
      "tokenizers_cpp_decoder_sequence_dispatch.json",
      tokenizer_with_slot(
          "decoder",
          R"json({"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"ByteFallback"}]})json"));

  const auto tokenizer = tokenizers_cpp::Tokenizer::from_file(write_tokenizer_json(
      "tokenizers_cpp_dispatch_preserves_wordlevel.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"WhitespaceSplit"})json")));
  const auto encoded = tokenizer.encode("hello world");
  assert((encoded.ids == std::vector<std::uint32_t>{1, 2}));
  assert((encoded.tokens == std::vector<std::string>{"hello", "world"}));

  expect_load_fails(
      "tokenizers_cpp_unsupported_pretok_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"NotAPreTokenizer"})json"),
      "unsupported pre_tokenizer wrapper type: NotAPreTokenizer");
  expect_load_fails(
      "tokenizers_cpp_unsupported_model_dispatch.json",
      tokenizer_with_model(R"json({"type":"NotAModel","vocab":{}})json"),
      "unsupported model wrapper type: NotAModel");
  expect_load_fails(
      "tokenizers_cpp_unsupported_normalizer_dispatch.json",
      tokenizer_with_slot("normalizer", R"json({"type":"NotANormalizer"})json"),
      "unsupported normalizer wrapper type: NotANormalizer");
  expect_load_fails(
      "tokenizers_cpp_unsupported_processor_dispatch.json",
      tokenizer_with_slot("post_processor", R"json({"type":"NotAProcessor"})json"),
      "unsupported post_processor wrapper type: NotAProcessor");
  expect_load_fails(
      "tokenizers_cpp_unsupported_decoder_dispatch.json",
      tokenizer_with_slot("decoder", R"json({"type":"NotADecoder"})json"),
      "unsupported decoder wrapper type: NotADecoder");
  expect_load_fails(
      "tokenizers_cpp_legacy_bert_normalizer_alias_dispatch.json",
      tokenizer_with_slot("normalizer", R"json({"type":"Bert"})json"),
      "unsupported normalizer wrapper type: Bert");
  expect_load_fails(
      "tokenizers_cpp_legacy_delimiter_alias_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"Delimiter","delimiter":" "})json"),
      "unsupported pre_tokenizer wrapper type: Delimiter");
  expect_load_fails(
      "tokenizers_cpp_wrong_slot_dispatch.json",
      tokenizer_with_slot("normalizer", R"json({"type":"BertPreTokenizer"})json"),
      "unsupported normalizer wrapper type: BertPreTokenizer");
  expect_load_fails(
      "tokenizers_cpp_sequence_missing_children_dispatch.json",
      tokenizer_with_slot("pre_tokenizer", R"json({"type":"Sequence"})json"),
      "Sequence is missing required field pretokenizers");
  expect_load_fails(
      "tokenizers_cpp_missing_model_dispatch.json",
      R"json({"version":"1.0"})json",
      "missing required model");

  std::filesystem::remove(std::filesystem::temp_directory_path() / "tokenizers_cpp_dispatch_preserves_wordlevel.json");
  return 0;
}
