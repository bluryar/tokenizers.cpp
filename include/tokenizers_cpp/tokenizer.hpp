#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tokenizers_cpp {

struct AddedToken {
  std::string content;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = true;
  bool special = false;
};

// Inference-time options for Tokenizer::from_bpe_files. This is the stable raw
// BPE file-loader surface; lower-level BPE builder/cache APIs remain private.
struct BpeOptions {
  std::optional<std::string> unk_token;
  std::optional<std::string> continuing_subword_prefix;
  std::optional<std::string> end_of_word_suffix;
  std::optional<double> dropout;
  bool fuse_unk = false;
  bool byte_fallback = false;
  bool ignore_merges = false;
};

// Offsets are byte offsets by default. Use the *_char_offsets APIs when
// downstream code needs Unicode scalar index offsets instead.
struct Offset {
  std::size_t start = 0;
  std::size_t end = 0;

  friend bool operator==(const Offset & lhs, const Offset & rhs) {
    return lhs.start == rhs.start && lhs.end == rhs.end;
  }
};

// Mirrors the Hugging Face inference fields used by downstream model runtimes.
struct Encoding {
  std::vector<std::uint32_t> ids;
  std::vector<std::uint32_t> type_ids;
  std::vector<std::string> tokens;
  std::vector<Offset> offsets;
  std::vector<std::optional<std::uint32_t>> word_ids;
  std::vector<std::uint32_t> special_tokens_mask;
  std::vector<std::uint32_t> attention_mask;
  std::vector<Encoding> overflowing;
};

class Tokenizer;

// Incremental decode helper. Incomplete byte fallback state is observable
// through has_pending(); callers reset by discarding the stream.
class DecodeStream {
public:
  std::optional<std::string> step(std::uint32_t id);
  bool has_pending() const;

private:
  friend class Tokenizer;

  DecodeStream(const Tokenizer & tokenizer, bool skip_special_tokens);

  const Tokenizer * tokenizer_ = nullptr;
  bool skip_special_tokens_ = true;
  std::string pending_;
  std::vector<std::uint32_t> ids_;
  std::string emitted_;
  std::string pending_byte_fallback_;
};

// Tokenizer-centered native C++ inference API.
class Tokenizer {
public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(const Tokenizer & other);
  Tokenizer & operator=(const Tokenizer & other);
  Tokenizer(Tokenizer && other) noexcept;
  Tokenizer & operator=(Tokenizer && other) noexcept;

  static Tokenizer from_file(const std::filesystem::path & path);
  static Tokenizer from_bpe_files(
      const std::filesystem::path & vocab_path,
      const std::filesystem::path & merges_path,
      const BpeOptions & options = {});

  Encoding encode(const std::string & text, bool add_special_tokens = true) const;
  // Encodes already pre-tokenized words. Offsets are relative to each input word,
  // matching Hugging Face pre-tokenized byte-offset semantics.
  Encoding encode(
      const std::vector<std::string> & pre_tokenized,
      bool add_special_tokens = true) const;
  Encoding encode_char_offsets(
      const std::string & text,
      bool add_special_tokens = true) const;
  Encoding encode_char_offsets(
      const std::vector<std::string> & pre_tokenized,
      bool add_special_tokens = true) const;
  Encoding encode_pair(
      const std::string & text_a,
      const std::string & text_b,
      bool add_special_tokens = true) const;
  Encoding encode_pair(
      const std::vector<std::string> & pre_tokenized_a,
      const std::vector<std::string> & pre_tokenized_b,
      bool add_special_tokens = true) const;
  Encoding encode_pair_char_offsets(
      const std::string & text_a,
      const std::string & text_b,
      bool add_special_tokens = true) const;
  Encoding encode_pair_char_offsets(
      const std::vector<std::string> & pre_tokenized_a,
      const std::vector<std::string> & pre_tokenized_b,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch(
      const std::vector<std::string> & texts,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch(
      const std::vector<std::vector<std::string>> & pre_tokenized_texts,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_char_offsets(
      const std::vector<std::string> & texts,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_char_offsets(
      const std::vector<std::vector<std::string>> & pre_tokenized_texts,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_pairs(
      const std::vector<std::pair<std::string, std::string>> & pairs,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_pairs(
      const std::vector<
          std::pair<std::vector<std::string>, std::vector<std::string>>> & pairs,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_pairs_char_offsets(
      const std::vector<std::pair<std::string, std::string>> & pairs,
      bool add_special_tokens = true) const;
  std::vector<Encoding> encode_batch_pairs_char_offsets(
      const std::vector<
          std::pair<std::vector<std::string>, std::vector<std::string>>> & pairs,
      bool add_special_tokens = true) const;

  std::string decode(
      const std::vector<std::uint32_t> & ids,
      bool skip_special_tokens = true) const;
  std::vector<std::string> decode_batch(
      const std::vector<std::vector<std::uint32_t>> & sequences,
      bool skip_special_tokens = true) const;
  DecodeStream decode_stream(bool skip_special_tokens = true) const;

  std::optional<std::uint32_t> token_to_id(const std::string & token) const;
  std::optional<std::string> id_to_token(std::uint32_t id) const;
  std::size_t get_vocab_size() const;

  std::size_t add_tokens(const std::vector<AddedToken> & tokens);
  void with_byte_level_normalizer();
  void with_split_pre_tokenizer(const std::string & regex_pattern);
  void with_wordpiece_decoder(
      std::string prefix = "##",
      bool cleanup = true);

private:
  friend class DecodeStream;
  struct Impl;

  Encoding encode_text(
      const std::string & text,
      bool add_special_tokens,
      bool apply_truncation,
      bool apply_padding) const;
  Encoding encode_pre_tokenized_words(
      const std::vector<std::string> & pre_tokenized,
      bool add_special_tokens,
      bool apply_truncation,
      bool apply_padding) const;
  Encoding encode_pair_text(
      const std::string & text_a,
      const std::string & text_b,
      bool add_special_tokens,
      bool apply_padding) const;
  Encoding encode_pair_pre_tokenized_words(
      const std::vector<std::string> & pre_tokenized_a,
      const std::vector<std::string> & pre_tokenized_b,
      bool add_special_tokens,
      bool apply_padding) const;

  std::unique_ptr<Impl> impl_;
};

}  // namespace tokenizers_cpp
