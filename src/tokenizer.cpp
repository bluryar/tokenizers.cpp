#include "tokenizers_cpp/tokenizer.hpp"
#include "tokenizers_cpp/unicode_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <functional>
#include <future>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>
#include <aho_corasick/aho_corasick.hpp>

namespace tokenizers_cpp {

template <
    typename Key,
    typename Value,
    typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
using HashMap = std::unordered_map<Key, Value, Hash, KeyEqual>;

namespace {

std::uint64_t allocate_private_cache_id();

}  // namespace

namespace detail {

struct AddedToken {
  std::uint32_t id = 0;
  std::string content;
  std::string normalized_content;
  bool single_word = false;
  bool lstrip = false;
  bool rstrip = false;
  bool normalized = true;
  bool special = false;
};

struct ByteLevelConfig {
  bool enabled = false;
  bool add_prefix_space = true;
  bool trim_offsets = true;
  bool use_regex = true;
};

struct SplitPreTokenizerConfig {
  bool enabled = false;
  bool llama_regex = false;
  bool llama_stream_escaped_regex = false;
  bool regex_pattern = true;
  std::string pattern;
  std::string behavior = "Isolated";
  bool invert = false;
};

enum class PreTokenizerStepKind {
  Split,
  Whitespace,
  Digits,
};

struct PreTokenizerStepConfig {
  PreTokenizerStepKind kind = PreTokenizerStepKind::Split;
  SplitPreTokenizerConfig split;
  bool individual_digits = false;
};

enum class SimpleNormalizerOpKind {
  Replace,
  Nfc,
  Nfd,
  Nfkc,
  Nfkd,
  Nmt,
  Prepend,
  Strip,
  StripAccents,
  Lowercase,
  Precompiled,
};

struct PrecompiledCharsMap {
  bool enabled = false;
  std::vector<std::uint32_t> trie;
  std::string normalized;
};

struct SimpleNormalizerOp {
  SimpleNormalizerOpKind kind = SimpleNormalizerOpKind::Replace;
  std::string pattern;
  std::string content;
  std::size_t precompiled_index = 0;
  bool regex_pattern = false;
  bool strip_left = true;
  bool strip_right = true;
};

struct SimpleNormalizerConfig {
  bool enabled = false;
  bool nfc = false;
  bool lowercase = false;
  bool nfd = false;
  bool nfkc = false;
  bool nfkd = false;
  bool nmt = false;
  bool prepend = false;
  bool strip = false;
  bool strip_accents = false;
  std::vector<std::pair<std::string, std::string>> replacements;
  std::vector<PrecompiledCharsMap> precompiled_maps;
  std::vector<SimpleNormalizerOp> ops;
};

struct MetaspaceConfig {
  bool enabled = false;
  std::string replacement = "\xE2\x96\x81";
  std::string prepend_scheme = "always";
  bool split = true;
};

enum class ProcessingPieceKind {
  SequenceA,
  SequenceB,
  Special,
};

struct ProcessingPiece {
  ProcessingPieceKind kind = ProcessingPieceKind::SequenceA;
  std::uint32_t type_id = 0;
  std::vector<std::uint32_t> ids;
  std::vector<std::string> tokens;
};

struct TemplateProcessingConfig {
  bool enabled = false;
  std::string sep_token = "[SEP]";
  std::uint32_t sep_id = 3;
  std::string cls_token = "[CLS]";
  std::uint32_t cls_id = 2;
  std::vector<ProcessingPiece> single_template;
  std::vector<ProcessingPiece> pair_template;
};

struct BertNormalizerConfig {
  bool enabled = false;
  bool clean_text = true;
  bool handle_chinese_chars = true;
  std::optional<bool> strip_accents;
  bool lowercase = true;
};

struct WordPieceConfig {
  struct TrieNode {
    std::optional<std::uint32_t> token_id;
    HashMap<unsigned char, std::size_t> children;
  };

  std::string unk_token = "[UNK]";
  std::string continuing_subword_prefix = "##";
  std::uint32_t max_input_chars_per_word = 100;
  std::optional<std::uint32_t> unk_id;
  std::vector<TrieNode> initial_trie;
  std::vector<TrieNode> continuation_trie;
};

struct WordPieceDecoderConfig {
  bool enabled = false;
  std::string prefix = "##";
  bool cleanup = true;
};

struct BpeDecoderConfig {
  std::string suffix = "</w>";
};

struct CtcDecoderConfig {
  std::string pad_token = "<pad>";
  std::string word_delimiter_token = "|";
  bool cleanup = true;
};

struct ReplaceDecoderConfig {
  std::string pattern;
  std::string content;
  bool regex_pattern = false;
};

struct StripDecoderConfig {
  std::string content;
  std::size_t start = 0;
  std::size_t stop = 0;
};

enum class DecoderStepKind {
  Bpe,
  ByteLevel,
  WordPiece,
  Metaspace,
  Ctc,
  Replace,
  Fuse,
  Strip,
  ByteFallback,
};

struct DecoderStepConfig {
  DecoderStepKind kind = DecoderStepKind::Fuse;
  BpeDecoderConfig bpe;
  ByteLevelConfig byte_level;
  WordPieceDecoderConfig wordpiece;
  MetaspaceConfig metaspace;
  CtcDecoderConfig ctc;
  ReplaceDecoderConfig replace;
  StripDecoderConfig strip;
};

enum class TruncationDirection {
  Right,
  Left,
};

enum class TruncationStrategy {
  LongestFirst,
  OnlyFirst,
  OnlySecond,
};

struct TruncationConfig {
  bool enabled = false;
  TruncationDirection direction = TruncationDirection::Right;
  TruncationStrategy strategy = TruncationStrategy::LongestFirst;
  std::size_t max_length = 512;
  std::size_t stride = 0;
};

enum class PaddingDirection {
  Right,
  Left,
};

enum class PaddingStrategy {
  BatchLongest,
  Fixed,
};

struct PaddingConfig {
  bool enabled = false;
  PaddingStrategy strategy = PaddingStrategy::BatchLongest;
  PaddingDirection direction = PaddingDirection::Right;
  std::optional<std::size_t> pad_to_multiple_of;
  std::size_t fixed_size = 0;
  std::uint32_t pad_id = 0;
  std::uint32_t pad_type_id = 0;
  std::string pad_token = "[PAD]";
};

struct WordLevelConfig {
  std::string unk_token = "<unk>";
  std::optional<std::uint32_t> unk_id;
};

struct BpeConfig {
  std::optional<std::string> unk_token;
  std::optional<std::uint32_t> unk_id;
  std::optional<std::string> continuing_subword_prefix;
  std::optional<std::string> end_of_word_suffix;
  std::optional<double> dropout;
  bool fuse_unk = false;
  bool byte_fallback = false;
  bool ignore_merges = false;
  std::array<std::optional<std::uint32_t>, 256> byte_fallback_ids{};
};

struct UnigramConfig {
  struct TrieNode {
    std::vector<std::uint32_t> token_ids;
    HashMap<unsigned char, std::size_t> children;
  };

  std::optional<std::uint32_t> unk_id;
  bool byte_fallback = false;
  bool fuse_unk = true;
  std::vector<double> scores;
  std::vector<TrieNode> trie;
  std::array<std::optional<std::uint32_t>, 256> byte_fallback_ids{};
  double min_score = 0.0;
};

struct BertProcessingConfig {
  bool enabled = false;
  std::string sep_token = "[SEP]";
  std::uint32_t sep_id = 102;
  std::string cls_token = "[CLS]";
  std::uint32_t cls_id = 101;
  std::vector<ProcessingPiece> single_template;
  std::vector<ProcessingPiece> pair_template;
  ByteLevelConfig offset_processor;
};

struct BpeMerge {
  std::uint32_t rank = 0;
  std::uint32_t new_id = 0;
};

}  // namespace detail

struct AddedTokenMatch {
  std::size_t token_index = 0;
  std::size_t start = 0;
  std::size_t stop = 0;
};

class AddedTokenMatcher {
 public:
  explicit AddedTokenMatcher(const std::vector<detail::AddedToken> & added_tokens) {
    token_indices_by_emit_.reserve(added_tokens.size());
    for (std::size_t token_index = 0; token_index < added_tokens.size(); ++token_index) {
      const auto & token = added_tokens[token_index];
      if (token.content.empty()) {
        continue;
      }
      trie_.insert(token.content);
      token_indices_by_emit_.push_back(token_index);
    }
    use_trie_ = token_indices_by_emit_.size() >= kTrieThreshold;
    if (use_trie_) {
      (void)trie_.parse_text(std::string{});
    }
  }

  bool use_trie() const {
    return use_trie_;
  }

  std::vector<AddedTokenMatch> find_matches(std::string_view text) {
    if (!use_trie_ || text.empty()) {
      return {};
    }

    std::vector<AddedTokenMatch> matches;
    for (const auto & emit : trie_.parse_text(std::string(text))) {
      const auto emit_index = static_cast<std::size_t>(emit.get_index());
      if (emit_index >= token_indices_by_emit_.size()) {
        continue;
      }
      matches.push_back(AddedTokenMatch{
          token_indices_by_emit_[emit_index],
          emit.get_start(),
          emit.get_end() + 1});
    }

    std::sort(
        matches.begin(),
        matches.end(),
        [](const AddedTokenMatch & lhs, const AddedTokenMatch & rhs) {
          if (lhs.start != rhs.start) {
            return lhs.start < rhs.start;
          }
          const auto lhs_length = lhs.stop - lhs.start;
          const auto rhs_length = rhs.stop - rhs.start;
          if (lhs_length != rhs_length) {
            return lhs_length > rhs_length;
          }
          return lhs.token_index < rhs.token_index;
        });
    return matches;
  }

 private:
  static constexpr std::size_t kTrieThreshold = 8;

  aho_corasick::trie trie_;
  std::vector<std::size_t> token_indices_by_emit_;
  bool use_trie_ = false;
};

struct Tokenizer::Impl {
  std::vector<std::string> id_to_token_;
  HashMap<std::string, std::uint32_t> token_to_id_;
  std::vector<bool> special_id_;
  std::vector<detail::AddedToken> added_tokens_;
  std::shared_ptr<AddedTokenMatcher> added_token_matcher_;
  std::string model_type_;
  HashMap<std::uint64_t, detail::BpeMerge> bpe_merges_;
  detail::BpeConfig bpe_;
  std::uint64_t bpe_cache_id_ = allocate_private_cache_id();
  detail::WordPieceConfig wordpiece_;
  detail::WordPieceDecoderConfig wordpiece_decoder_;
  detail::WordLevelConfig wordlevel_;
  detail::UnigramConfig unigram_;
  std::uint64_t unigram_cache_id_ = allocate_private_cache_id();
  bool byte_level_normalizer_ = false;
  detail::SimpleNormalizerConfig simple_normalizer_;
  detail::BertNormalizerConfig bert_normalizer_;
  bool bert_pre_tokenizer_ = false;
  bool whitespace_pre_tokenizer_ = false;
  detail::SplitPreTokenizerConfig split_pre_tokenizer_;
  std::vector<detail::PreTokenizerStepConfig> pre_tokenizer_steps_;
  detail::BertProcessingConfig bert_processing_;
  detail::TemplateProcessingConfig template_processing_;
  detail::MetaspaceConfig metaspace_pre_tokenizer_;
  bool whitespace_split_pre_tokenizer_ = false;
  detail::ByteLevelConfig byte_level_pre_tokenizer_;
  detail::ByteLevelConfig byte_level_post_processor_;
  detail::ByteLevelConfig byte_level_decoder_;
  detail::MetaspaceConfig metaspace_decoder_;
  std::vector<detail::DecoderStepConfig> decoder_steps_;
  detail::TruncationConfig truncation_;
  detail::PaddingConfig padding_;
};

namespace {

using json = nlohmann::json;

std::atomic<std::uint64_t> g_next_private_cache_id{1};

std::uint64_t allocate_private_cache_id() {
  return g_next_private_cache_id.fetch_add(1, std::memory_order_relaxed);
}

std::size_t next_codepoint(std::string_view text, std::size_t pos);

template <typename Output, typename EncodeOne>
std::vector<Output> map_batch_ordered(
    std::size_t size,
    std::size_t work_units,
    EncodeOne encode_one) {
  static constexpr std::size_t kParallelMinItems = 8;
  static constexpr std::size_t kParallelMinWorkUnits = 8192;
  static constexpr std::size_t kParallelItemsPerThread = 8;
  static constexpr std::size_t kParallelWorkUnitsPerThread = 4096;

  std::vector<Output> outputs(size);
  if (size == 0) {
    return outputs;
  }

  const auto hardware_threads =
      static_cast<std::size_t>(std::max(1U, std::thread::hardware_concurrency()));
  const auto item_limited_threads =
      std::max<std::size_t>(1, size / kParallelItemsPerThread);
  const auto work_limited_threads = std::max<std::size_t>(
      1,
      work_units / kParallelWorkUnitsPerThread);
  const auto worker_count =
      std::min(
          {size,
           hardware_threads,
           std::max(item_limited_threads, work_limited_threads)});
  if (worker_count <= 1 ||
      (size < kParallelMinItems && work_units < kParallelMinWorkUnits)) {
    for (std::size_t index = 0; index < size; ++index) {
      outputs[index] = encode_one(index);
    }
    return outputs;
  }

  std::vector<std::future<void>> futures;
  futures.reserve(worker_count);
  const auto chunk_size = (size + worker_count - 1) / worker_count;
  for (std::size_t begin = 0; begin < size; begin += chunk_size) {
    const auto end = std::min(begin + chunk_size, size);
    futures.push_back(std::async(
        std::launch::async,
        [begin, end, &outputs, &encode_one] {
          for (std::size_t index = begin; index < end; ++index) {
            outputs[index] = encode_one(index);
          }
        }));
  }
  for (auto & future : futures) {
    future.get();
  }
  return outputs;
}

std::size_t total_text_bytes(const std::vector<std::string> & texts) {
  std::size_t total = 0;
  for (const auto & text : texts) {
    total += text.size();
  }
  return total;
}

std::size_t total_text_bytes(
    const std::vector<std::vector<std::string>> & pre_tokenized_texts) {
  std::size_t total = 0;
  for (const auto & words : pre_tokenized_texts) {
    total += total_text_bytes(words);
  }
  return total;
}

std::size_t total_pair_text_bytes(
    const std::vector<std::pair<std::string, std::string>> & pairs) {
  std::size_t total = 0;
  for (const auto & pair : pairs) {
    total += pair.first.size() + pair.second.size();
  }
  return total;
}

std::size_t total_pair_text_bytes(
    const std::vector<
        std::pair<std::vector<std::string>, std::vector<std::string>>> & pairs) {
  std::size_t total = 0;
  for (const auto & pair : pairs) {
    total += total_text_bytes(pair.first) + total_text_bytes(pair.second);
  }
  return total;
}

std::size_t total_id_count(
    const std::vector<std::vector<std::uint32_t>> & sequences) {
  std::size_t total = 0;
  for (const auto & sequence : sequences) {
    total += sequence.size();
  }
  return total * sizeof(std::uint32_t);
}

std::size_t byte_offset_to_char_offset(std::string_view text, std::size_t byte_offset) {
  std::size_t char_index = 0;
  for (std::size_t pos = 0; pos < text.size();) {
    const auto end = next_codepoint(text, pos);
    if (byte_offset < end) {
      return char_index;
    }
    if (byte_offset == end) {
      return char_index + 1;
    }
    pos = end;
    ++char_index;
  }
  return char_index;
}

Offset byte_offsets_to_char_offsets(std::string_view text, Offset offsets) {
  return Offset{
      byte_offset_to_char_offset(text, offsets.start),
      byte_offset_to_char_offset(text, offsets.end),
  };
}

template <typename SourceForToken>
void convert_encoding_offsets_to_char_offsets(
    Encoding & encoding,
    SourceForToken source_for_token) {
  for (std::size_t index = 0; index < encoding.offsets.size(); ++index) {
    const auto source = source_for_token(encoding, index);
    if (source) {
      encoding.offsets[index] =
          byte_offsets_to_char_offsets(*source, encoding.offsets[index]);
    }
  }
  for (auto & overflow : encoding.overflowing) {
    convert_encoding_offsets_to_char_offsets(overflow, source_for_token);
  }
}

void convert_offsets_to_char_offsets(Encoding & encoding, std::string_view text) {
  convert_encoding_offsets_to_char_offsets(
      encoding,
      [text](const Encoding &, std::size_t) -> std::optional<std::string_view> {
        return text;
      });
}

void convert_pair_offsets_to_char_offsets(
    Encoding & encoding,
    std::string_view text_a,
    std::string_view text_b) {
  convert_encoding_offsets_to_char_offsets(
      encoding,
      [text_a, text_b](
          const Encoding & current,
          std::size_t index) -> std::optional<std::string_view> {
        if (index < current.type_ids.size() && current.type_ids[index] == 1) {
          return text_b;
        }
        return text_a;
      });
}

void convert_pre_tokenized_offsets_to_char_offsets(
    Encoding & encoding,
    const std::vector<std::string> & pre_tokenized) {
  convert_encoding_offsets_to_char_offsets(
      encoding,
      [&pre_tokenized](
          const Encoding & current,
          std::size_t index) -> std::optional<std::string_view> {
        if (index >= current.word_ids.size() || !current.word_ids[index]) {
          return std::nullopt;
        }
        const auto word_id = static_cast<std::size_t>(*current.word_ids[index]);
        if (word_id >= pre_tokenized.size()) {
          return std::nullopt;
        }
        return std::string_view(pre_tokenized[word_id]);
      });
}

void convert_pre_tokenized_pair_offsets_to_char_offsets(
    Encoding & encoding,
    const std::vector<std::string> & pre_tokenized_a,
    const std::vector<std::string> & pre_tokenized_b) {
  convert_encoding_offsets_to_char_offsets(
      encoding,
      [&pre_tokenized_a, &pre_tokenized_b](
          const Encoding & current,
          std::size_t index) -> std::optional<std::string_view> {
        if (index >= current.word_ids.size() || !current.word_ids[index]) {
          return std::nullopt;
        }
        const auto & words = index < current.type_ids.size() &&
                current.type_ids[index] == 1
            ? pre_tokenized_b
            : pre_tokenized_a;
        const auto word_id = static_cast<std::size_t>(*current.word_ids[index]);
        if (word_id >= words.size()) {
          return std::nullopt;
        }
        return std::string_view(words[word_id]);
      });
}

void ensure_id(
    std::vector<std::string> & id_to_token,
    std::vector<bool> & special_id,
    std::uint32_t id) {
  if (id >= id_to_token.size()) {
    id_to_token.resize(static_cast<std::size_t>(id) + 1);
    special_id.resize(static_cast<std::size_t>(id) + 1, false);
  }
}

void rebuild_token_to_id(
    const std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id) {
  token_to_id.clear();
  token_to_id.reserve(id_to_token.size());
  for (std::uint32_t id = 0; id < id_to_token.size(); ++id) {
    if (!id_to_token[id].empty()) {
      token_to_id.emplace(id_to_token[id], id);
    }
  }
}

void load_vocab_entry(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const std::string & token,
    std::uint32_t id,
    bool special = false) {
  ensure_id(id_to_token, special_id, id);
  const auto & previous_token = id_to_token[id];
  if (!previous_token.empty() && previous_token != token) {
    const auto previous = token_to_id.find(previous_token);
    if (previous != token_to_id.end() && previous->second == id) {
      token_to_id.erase(previous);
    }
  }
  id_to_token[id] = token;
  if (!token.empty()) {
    token_to_id.emplace(token, id);
  }
  special_id[id] = special_id[id] || special;
}

void load_vocab_object(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const json & vocab) {
  token_to_id.reserve(token_to_id.size() + vocab.size());
  for (const auto & item : vocab.items()) {
    load_vocab_entry(
        id_to_token,
        token_to_id,
        special_id,
        item.key(),
        item.value().get<std::uint32_t>());
  }
}

void load_vocab_array(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const json & vocab) {
  token_to_id.reserve(token_to_id.size() + vocab.size());
  for (std::uint32_t id = 0; id < vocab.size(); ++id) {
    const auto & entry = vocab.at(id);
    if (entry.is_array() && !entry.empty() && entry.at(0).is_string()) {
      load_vocab_entry(
          id_to_token,
          token_to_id,
          special_id,
          entry.at(0).get<std::string>(),
          id);
    } else if (entry.is_object() && entry.contains("token")) {
      const auto token = entry.at("token").get<std::string>();
      const auto entry_id = entry.value("id", id);
      load_vocab_entry(id_to_token, token_to_id, special_id, token, entry_id);
    }
  }
}

std::vector<std::pair<std::string, Offset>> split_on_ascii_whitespace(
    const std::string & text) {
  std::vector<std::pair<std::string, Offset>> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    const std::size_t start = pos;
    while (pos < text.size() && !std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (start < pos) {
      pieces.push_back({text.substr(start, pos - start), Offset{start, pos}});
    }
  }
  return pieces;
}

bool is_ascii_space(char value) {
  return std::isspace(static_cast<unsigned char>(value)) != 0;
}

bool is_ascii_word(char value) {
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '_';
}

bool has_word_byte_before(std::string_view text, std::size_t pos) {
  return pos > 0 && is_ascii_word(text[pos - 1]);
}

bool has_word_byte_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_ascii_word(text[pos]);
}

std::size_t whitespace_start_before(std::string_view text, std::size_t pos) {
  while (pos > 0 && is_ascii_space(text[pos - 1])) {
    --pos;
  }
  return pos;
}

std::size_t whitespace_end_after(std::string_view text, std::size_t pos) {
  while (pos < text.size() && is_ascii_space(text[pos])) {
    ++pos;
  }
  return pos;
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
      value.substr(0, prefix.size()) == prefix;
}

std::uint64_t bpe_pair_key(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

std::size_t utf8_codepoint_length(unsigned char lead) {
  if ((lead & 0x80U) == 0) {
    return 1;
  }
  if ((lead & 0xE0U) == 0xC0U) {
    return 2;
  }
  if ((lead & 0xF0U) == 0xE0U) {
    return 3;
  }
  if ((lead & 0xF8U) == 0xF0U) {
    return 4;
  }
  return 1;
}

std::string utf8_from_codepoint(std::uint32_t codepoint) {
  std::string out;
  if (codepoint <= 0x7FU) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFU) {
    out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else if (codepoint <= 0xFFFFU) {
    out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  } else {
    out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
  }
  return out;
}

const std::vector<std::string> & byte_to_unicode() {
  static const std::vector<std::string> table = [] {
    std::vector<unsigned char> bytes;
    for (int value = '!'; value <= '~'; ++value) {
      bytes.push_back(static_cast<unsigned char>(value));
    }
    for (int value = 0xA1; value <= 0xAC; ++value) {
      bytes.push_back(static_cast<unsigned char>(value));
    }
    for (int value = 0xAE; value <= 0xFF; ++value) {
      bytes.push_back(static_cast<unsigned char>(value));
    }

    std::vector<std::uint32_t> chars;
    chars.reserve(256);
    for (const auto byte : bytes) {
      chars.push_back(byte);
    }

    std::uint32_t extra = 0;
    for (int value = 0; value <= 0xFF; ++value) {
      const auto byte = static_cast<unsigned char>(value);
      if (std::find(bytes.begin(), bytes.end(), byte) == bytes.end()) {
        bytes.push_back(byte);
        chars.push_back(256U + extra);
        ++extra;
      }
    }

    std::vector<std::string> result(256);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      result[bytes[index]] = utf8_from_codepoint(chars[index]);
    }
    return result;
  }();
  return table;
}

const std::string & byte_level_space() {
  static const std::string value = byte_to_unicode()[static_cast<unsigned char>(' ')];
  return value;
}

const HashMap<std::string, unsigned char> & unicode_to_byte() {
  static const HashMap<std::string, unsigned char> table = [] {
    HashMap<std::string, unsigned char> result;
    const auto & byte_map = byte_to_unicode();
    for (std::size_t byte = 0; byte < byte_map.size(); ++byte) {
      result.emplace(byte_map[byte], static_cast<unsigned char>(byte));
    }
    return result;
  }();
  return table;
}

std::string decode_byte_level_tokens(const std::vector<std::string> & tokens) {
  std::string bytes;
  const auto & byte_map = unicode_to_byte();
  for (const auto & token : tokens) {
    std::string token_bytes;
    bool mapped = true;
    for (std::size_t pos = 0; pos < token.size();) {
      const auto length = utf8_codepoint_length(static_cast<unsigned char>(token[pos]));
      const auto current = token.substr(pos, length);
      const auto found = byte_map.find(current);
      if (found == byte_map.end()) {
        mapped = false;
        break;
      }
      token_bytes.push_back(static_cast<char>(found->second));
      pos += length;
    }
    bytes.append(mapped ? token_bytes : token);
  }
  return bytes;
}

std::size_t valid_utf8_prefix_size(std::string_view bytes) {
  std::size_t pos = 0;
  std::size_t last_valid = 0;
  while (pos < bytes.size()) {
    const auto lead = static_cast<unsigned char>(bytes[pos]);
    std::size_t length = 0;
    if ((lead & 0x80U) == 0) {
      length = 1;
    } else if ((lead & 0xE0U) == 0xC0U) {
      length = 2;
    } else if ((lead & 0xF0U) == 0xE0U) {
      length = 3;
    } else if ((lead & 0xF8U) == 0xF0U) {
      length = 4;
    } else {
      return last_valid;
    }

    if (pos + length > bytes.size()) {
      break;
    }
    for (std::size_t index = 1; index < length; ++index) {
      const auto value = static_cast<unsigned char>(bytes[pos + index]);
      if ((value & 0xC0U) != 0x80U) {
        return last_valid;
      }
    }
    pos += length;
    last_valid = pos;
  }
  return last_valid;
}

std::string decode_metaspace_tokens(
    const std::vector<std::string> & tokens,
    const detail::MetaspaceConfig & config) {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());
  for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
    const auto & token = tokens[token_index];
    std::string decoded;
    for (std::size_t pos = 0; pos < token.size();) {
      if (!config.replacement.empty() &&
          pos + config.replacement.size() <= token.size() &&
          token.compare(pos, config.replacement.size(), config.replacement) == 0) {
        if (!(token_index == 0 && config.prepend_scheme != "never")) {
          decoded.push_back(' ');
        }
        pos += config.replacement.size();
        continue;
      }
      const auto end = next_codepoint(token, pos);
      decoded.append(token.substr(pos, end - pos));
      pos = end;
    }
    decoded_tokens.push_back(std::move(decoded));
  }

  std::string output;
  for (const auto & token : decoded_tokens) {
    output.append(token);
  }
  return output;
}

void replace_all(std::string & value, std::string_view pattern, std::string_view replacement) {
  if (pattern.empty()) {
    return;
  }
  std::size_t pos = 0;
  while ((pos = value.find(pattern, pos)) != std::string::npos) {
    value.replace(pos, pattern.size(), replacement);
    pos += replacement.size();
  }
}

std::string cleanup_wordpiece_token(std::string token) {
  replace_all(token, " .", ".");
  replace_all(token, " ?", "?");
  replace_all(token, " !", "!");
  replace_all(token, " ,", ",");
  replace_all(token, " ' ", "'");
  replace_all(token, " n't", "n't");
  replace_all(token, " 'm", "'m");
  replace_all(token, " do not", " don't");
  replace_all(token, " 's", "'s");
  replace_all(token, " 've", "'ve");
  replace_all(token, " 're", "'re");
  return token;
}

std::string decode_wordpiece_tokens(
    std::vector<std::string> tokens,
    const detail::WordPieceDecoderConfig & config) {
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    auto & token = tokens[index];
    if (index != 0) {
      if (starts_with(token, config.prefix)) {
        token.erase(0, config.prefix.size());
      } else {
        token.insert(token.begin(), ' ');
      }
    }
    if (config.cleanup) {
      token = cleanup_wordpiece_token(std::move(token));
    }
  }

  std::string output;
  for (const auto & token : tokens) {
    output.append(token);
  }
  return output;
}

std::string join_decoded_tokens(const std::vector<std::string> & tokens) {
  std::string output;
  for (const auto & token : tokens) {
    output.append(token);
  }
  return output;
}

std::string join_decoded_tokens_with_spaces(
    const std::vector<std::string> & tokens) {
  std::string output;
  for (const auto & token : tokens) {
    if (!output.empty()) {
      output.push_back(' ');
    }
    output.append(token);
  }
  return output;
}

std::vector<std::string> decode_metaspace_chain(
    const std::vector<std::string> & tokens,
    const detail::MetaspaceConfig & config) {
  std::vector<std::string> output;
  output.reserve(tokens.size());
  for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
    const auto & token = tokens[token_index];
    std::string decoded;
    for (std::size_t pos = 0; pos < token.size();) {
      if (!config.replacement.empty() &&
          pos + config.replacement.size() <= token.size() &&
          token.compare(pos, config.replacement.size(), config.replacement) == 0) {
        if (!(token_index == 0 && config.prepend_scheme != "never")) {
          decoded.push_back(' ');
        }
        pos += config.replacement.size();
        continue;
      }
      const auto end = next_codepoint(token, pos);
      decoded.append(token.substr(pos, end - pos));
      pos = end;
    }
    output.push_back(std::move(decoded));
  }
  return output;
}

std::vector<std::string> decode_wordpiece_chain(
    std::vector<std::string> tokens,
    const detail::WordPieceDecoderConfig & config) {
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    auto & token = tokens[index];
    if (index != 0) {
      if (starts_with(token, config.prefix)) {
        token.erase(0, config.prefix.size());
      } else {
        token.insert(token.begin(), ' ');
      }
    }
    if (config.cleanup) {
      token = cleanup_wordpiece_token(std::move(token));
    }
  }
  return tokens;
}

std::optional<unsigned char> byte_fallback_token_value(const std::string & token) {
  if (token.size() != 6 || !starts_with(token, "<0x") || token.back() != '>') {
    return std::nullopt;
  }
  auto hex_value = [](char value) -> std::optional<unsigned char> {
    if (value >= '0' && value <= '9') {
      return static_cast<unsigned char>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<unsigned char>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
      return static_cast<unsigned char>(value - 'A' + 10);
    }
    return std::nullopt;
  };
  const auto high = hex_value(token[3]);
  const auto low = hex_value(token[4]);
  if (!high || !low) {
    return std::nullopt;
  }
  return static_cast<unsigned char>((*high << 4U) | *low);
}

void flush_byte_fallback_run(
    std::vector<std::string> & output,
    std::string & pending_bytes) {
  if (pending_bytes.empty()) {
    return;
  }
  if (valid_utf8_prefix_size(pending_bytes) == pending_bytes.size()) {
    output.push_back(pending_bytes);
  } else {
    for (std::size_t index = 0; index < pending_bytes.size(); ++index) {
      output.push_back("\xEF\xBF\xBD");
    }
  }
  pending_bytes.clear();
}

std::vector<std::string> decode_byte_fallback_chain(
    const std::vector<std::string> & tokens) {
  std::vector<std::string> output;
  std::string pending_bytes;
  for (const auto & token : tokens) {
    if (const auto byte = byte_fallback_token_value(token)) {
      pending_bytes.push_back(static_cast<char>(*byte));
      continue;
    }
    flush_byte_fallback_run(output, pending_bytes);
    output.push_back(token);
  }
  flush_byte_fallback_run(output, pending_bytes);
  return output;
}

std::vector<std::string> decode_fuse_chain(const std::vector<std::string> & tokens) {
  return {join_decoded_tokens(tokens)};
}

std::vector<std::string> decode_bpe_chain(
    std::vector<std::string> tokens,
    const detail::BpeDecoderConfig & config) {
  if (tokens.empty()) {
    return tokens;
  }
  const auto last = tokens.size() - 1;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    replace_all(tokens[index], config.suffix, index == last ? "" : " ");
  }
  return tokens;
}

std::vector<std::string> decode_replace_chain(
    std::vector<std::string> tokens,
    const detail::ReplaceDecoderConfig & config) {
  for (auto & token : tokens) {
    if (config.regex_pattern) {
      std::string output;
      std::size_t pos = 0;
      for (const auto & match : unicode::find_regex_matches_utf8(token, config.pattern)) {
        if (match.start > pos) {
          output.append(token.substr(pos, match.start - pos));
        }
        if (match.start < match.end) {
          output.append(config.content);
        }
        pos = std::max(pos, match.end);
      }
      if (pos < token.size()) {
        output.append(token.substr(pos));
      }
      token = std::move(output);
    } else {
      replace_all(token, config.pattern, config.content);
    }
  }
  return tokens;
}

std::vector<std::string> token_codepoints(std::string_view token) {
  std::vector<std::string> codepoints;
  for (std::size_t pos = 0; pos < token.size();) {
    const auto end = next_codepoint(token, pos);
    codepoints.emplace_back(token.substr(pos, end - pos));
    pos = end;
  }
  return codepoints;
}

std::vector<std::string> decode_strip_chain(
    const std::vector<std::string> & tokens,
    const detail::StripDecoderConfig & config) {
  std::vector<std::string> output;
  output.reserve(tokens.size());
  for (const auto & token : tokens) {
    const auto codepoints = token_codepoints(token);
    std::size_t start = 0;
    for (std::size_t index = 0; index < config.start && index < codepoints.size(); ++index) {
      if (codepoints[index] != config.content) {
        break;
      }
      start = index + 1;
    }

    std::size_t stop = codepoints.size();
    for (std::size_t index = 0; index < config.stop && index < codepoints.size(); ++index) {
      const auto current = codepoints.size() - index - 1;
      if (codepoints[current] != config.content) {
        break;
      }
      stop = current;
    }

    std::string stripped;
    for (std::size_t index = start; index < stop; ++index) {
      stripped.append(codepoints[index]);
    }
    output.push_back(std::move(stripped));
  }
  return output;
}

std::vector<std::string> decode_ctc_chain(
    const std::vector<std::string> & tokens,
    const detail::CtcDecoderConfig & config) {
  std::vector<std::string> output;
  std::optional<std::string> previous;
  for (const auto & token : tokens) {
    if (previous && *previous == token) {
      continue;
    }
    previous = token;

    auto decoded = token;
    replace_all(decoded, config.pad_token, "");
    if (config.cleanup) {
      decoded = cleanup_wordpiece_token(decoded);
      replace_all(decoded, config.word_delimiter_token, " ");
    }
    if (!decoded.empty()) {
      output.push_back(std::move(decoded));
    }
  }
  return output;
}

std::vector<std::string> apply_decoder_steps(
    std::vector<std::string> tokens,
    const std::vector<detail::DecoderStepConfig> & steps) {
  for (const auto & step : steps) {
    switch (step.kind) {
      case detail::DecoderStepKind::Bpe:
        tokens = decode_bpe_chain(std::move(tokens), step.bpe);
        break;
      case detail::DecoderStepKind::ByteLevel:
        tokens = {decode_byte_level_tokens(tokens)};
        break;
      case detail::DecoderStepKind::WordPiece:
        tokens = decode_wordpiece_chain(std::move(tokens), step.wordpiece);
        break;
      case detail::DecoderStepKind::Metaspace:
        tokens = decode_metaspace_chain(tokens, step.metaspace);
        break;
      case detail::DecoderStepKind::Ctc:
        tokens = decode_ctc_chain(tokens, step.ctc);
        break;
      case detail::DecoderStepKind::Replace:
        tokens = decode_replace_chain(std::move(tokens), step.replace);
        break;
      case detail::DecoderStepKind::Fuse:
        tokens = decode_fuse_chain(tokens);
        break;
      case detail::DecoderStepKind::Strip:
        tokens = decode_strip_chain(tokens, step.strip);
        break;
      case detail::DecoderStepKind::ByteFallback:
        tokens = decode_byte_fallback_chain(tokens);
        break;
    }
  }
  return tokens;
}

struct InputSplit {
  bool is_added_token = false;
  std::uint32_t id = 0;
  std::string text;
  Offset offset;
  std::string token_text;
};

struct OriginalSpan {
  std::size_t start = 0;
  std::size_t end = 0;
};

struct RawPiece {
  std::size_t start = 0;
  std::size_t end = 0;
};

struct SplitSpan {
  std::size_t start = 0;
  std::size_t end = 0;
  bool is_match = false;
};

struct ByteLevelPiece {
  std::string text;
  std::vector<OriginalSpan> normalized_byte_spans;
};

struct BpeSymbol {
  std::uint32_t id = 0;
  std::size_t start = 0;
  std::size_t end = 0;
};

struct BpeMergeNode {
  BpeSymbol symbol;
  std::optional<std::size_t> previous;
  std::optional<std::size_t> next;
  std::uint32_t generation = 0;
  bool active = true;
};

struct BpeMergeCandidate {
  std::uint32_t rank = 0;
  std::size_t left = 0;
  std::uint32_t left_generation = 0;
  std::size_t right = 0;
  std::uint32_t right_generation = 0;
  detail::BpeMerge merge;
};

struct BpeMergeCandidateCompare {
  bool operator()(
      const BpeMergeCandidate & lhs,
      const BpeMergeCandidate & rhs) const {
    if (lhs.rank != rhs.rank) {
      return lhs.rank > rhs.rank;
    }
    return lhs.left > rhs.left;
  }
};

struct BpeToken {
  std::uint32_t id = 0;
  std::string value;
  Offset offset;
};

using BpeCachedSymbols = std::vector<BpeSymbol>;
using BpeThreadCache = HashMap<std::string, BpeCachedSymbols>;
using UnigramCachedTokens = std::vector<BpeToken>;
using UnigramThreadCache = HashMap<std::string, UnigramCachedTokens>;

static constexpr std::size_t kBpeCacheCapacity = 10000;
static constexpr std::size_t kBpeCacheMaxLength = 256;
static constexpr std::size_t kUnigramCacheCapacity = 10000;
static constexpr std::size_t kUnigramCacheMaxLength = 256;

thread_local HashMap<std::uint64_t, BpeThreadCache> bpe_thread_cache;
thread_local HashMap<std::uint64_t, UnigramThreadCache>
    unigram_thread_cache;

struct UnigramBestPathNode {
  std::uint32_t id = 0;
  double score = 0.0;
  std::optional<std::size_t> starts_at;
};

struct UnigramSpan {
  std::uint32_t id = 0;
  std::size_t start = 0;
  std::size_t end = 0;
  bool is_unk = false;
};

std::string bpe_byte_fallback_token(unsigned char byte) {
  static constexpr char hex[] = "0123456789ABCDEF";
  std::string token = "<0x";
  token.push_back(hex[(byte >> 4U) & 0x0FU]);
  token.push_back(hex[byte & 0x0FU]);
  token.push_back('>');
  return token;
}

bool should_skip_bpe_merge(std::optional<double> dropout) {
  if (!dropout || *dropout <= 0.0) {
    return false;
  }
  if (*dropout >= 1.0) {
    return true;
  }
  thread_local std::mt19937 rng(std::random_device{}());
  std::bernoulli_distribution distribution(*dropout);
  return distribution(rng);
}

void flush_pending_unk(
    std::vector<BpeSymbol> & symbols,
    std::optional<BpeSymbol> & pending_unk) {
  if (pending_unk) {
    symbols.push_back(*pending_unk);
    pending_unk.reset();
  }
}

struct NormalizedInput {
  std::string text;
  std::vector<OriginalSpan> normalized_byte_spans;
};

NormalizedInput normalize_unicode_normalized(
    const NormalizedInput & input,
    unicode::NormalizationForm form);
NormalizedInput strip_normalized(
    const NormalizedInput & input,
    bool strip_left,
    bool strip_right);
NormalizedInput nmt_normalized(const NormalizedInput & input);
NormalizedInput prepend_normalized(
    const NormalizedInput & input,
    const std::string & prepend);
NormalizedInput strip_accents_normalized(const NormalizedInput & input);
NormalizedInput lowercase_unicode_normalized(const NormalizedInput & input);

struct NormalizedPiece {
  std::string text;
  std::vector<OriginalSpan> normalized_byte_spans;
};

struct PreTokenPiece {
  std::size_t start = 0;
  std::size_t end = 0;
};

bool is_ascii_letter_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return false;
  }
  const auto value = static_cast<unsigned char>(text[pos]);
  return value < 0x80U && std::isalpha(value) != 0;
}

bool is_ascii_number_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return false;
  }
  const auto value = static_cast<unsigned char>(text[pos]);
  return value < 0x80U && std::isdigit(value) != 0;
}

bool is_ascii_space_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return false;
  }
  const auto value = static_cast<unsigned char>(text[pos]);
  return value < 0x80U && std::isspace(value) != 0;
}

bool is_other_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && !is_ascii_space_at(text, pos) &&
      !is_ascii_letter_at(text, pos) && !is_ascii_number_at(text, pos);
}

std::size_t next_codepoint(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return pos;
  }
  const auto length = utf8_codepoint_length(static_cast<unsigned char>(text[pos]));
  return std::min(text.size(), pos + length);
}

std::size_t previous_codepoint_start(std::string_view text, std::size_t end) {
  if (end == 0) {
    return 0;
  }
  --end;
  while (end > 0 &&
      (static_cast<unsigned char>(text[end]) & 0xC0U) == 0x80U) {
    --end;
  }
  return end;
}

std::uint32_t utf8_codepoint_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return 0;
  }

  const auto lead = static_cast<unsigned char>(text[pos]);
  if ((lead & 0x80U) == 0) {
    return lead;
  }
  if ((lead & 0xE0U) == 0xC0U && pos + 1 < text.size()) {
    return ((lead & 0x1FU) << 6U) |
        (static_cast<unsigned char>(text[pos + 1]) & 0x3FU);
  }
  if ((lead & 0xF0U) == 0xE0U && pos + 2 < text.size()) {
    return ((lead & 0x0FU) << 12U) |
        ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 6U) |
        (static_cast<unsigned char>(text[pos + 2]) & 0x3FU);
  }
  if ((lead & 0xF8U) == 0xF0U && pos + 3 < text.size()) {
    return ((lead & 0x07U) << 18U) |
        ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 12U) |
        ((static_cast<unsigned char>(text[pos + 2]) & 0x3FU) << 6U) |
        (static_cast<unsigned char>(text[pos + 3]) & 0x3FU);
  }
  return lead;
}

bool is_chinese_codepoint(std::uint32_t codepoint) {
  return (codepoint >= 0x4E00U && codepoint <= 0x9FFFU) ||
      (codepoint >= 0x3400U && codepoint <= 0x4DBFU) ||
      (codepoint >= 0x20000U && codepoint <= 0x2A6DFU) ||
      (codepoint >= 0x2A700U && codepoint <= 0x2B73FU) ||
      (codepoint >= 0x2B740U && codepoint <= 0x2B81FU) ||
      (codepoint >= 0x2B820U && codepoint <= 0x2CEAFU) ||
      (codepoint >= 0xF900U && codepoint <= 0xFAFFU) ||
      (codepoint >= 0x2F800U && codepoint <= 0x2FA1FU);
}

bool is_bert_control_ascii(unsigned char value) {
  if (value == '\t' || value == '\n' || value == '\r') {
    return false;
  }
  return value < 0x20U || value == 0x7FU;
}

bool is_bert_whitespace_codepoint(std::uint32_t codepoint) {
  return codepoint == '\t' ||
      codepoint == '\n' ||
      codepoint == '\r' ||
      unicode::is_whitespace(codepoint);
}

bool is_line_break_codepoint(std::uint32_t codepoint) {
  return codepoint == '\r' || codepoint == '\n';
}

bool is_llama_number_codepoint(std::uint32_t codepoint) {
  return unicode::is_number(codepoint);
}

bool is_llama_letter_codepoint(std::uint32_t codepoint) {
  return unicode::is_letter(codepoint);
}

bool is_llama_space_codepoint(std::uint32_t codepoint) {
  return is_bert_whitespace_codepoint(codepoint);
}

bool is_llama_other_codepoint(std::uint32_t codepoint) {
  return !is_llama_space_codepoint(codepoint) &&
      !is_llama_letter_codepoint(codepoint) &&
      !is_llama_number_codepoint(codepoint);
}

bool is_llama_letter_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_llama_letter_codepoint(utf8_codepoint_at(text, pos));
}

bool is_llama_number_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_llama_number_codepoint(utf8_codepoint_at(text, pos));
}

bool is_llama_space_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_llama_space_codepoint(utf8_codepoint_at(text, pos));
}

bool is_llama_line_break_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_line_break_codepoint(utf8_codepoint_at(text, pos));
}

bool is_llama_other_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && is_llama_other_codepoint(utf8_codepoint_at(text, pos));
}

bool is_bert_control_codepoint(std::uint32_t codepoint) {
  if (codepoint == '\t' || codepoint == '\n' || codepoint == '\r') {
    return false;
  }
  return codepoint == 0U ||
      codepoint == 0xFFFDU ||
      unicode::is_control(codepoint);
}

void append_normalized_bytes(
    NormalizedInput & normalized,
    std::string_view bytes,
    OriginalSpan original_span) {
  normalized.text.append(bytes);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    normalized.normalized_byte_spans.push_back(original_span);
  }
}

NormalizedInput identity_normalized_input(const InputSplit & split) {
  NormalizedInput normalized;
  for (std::size_t pos = 0; pos < split.text.size();) {
    const auto end = next_codepoint(split.text, pos);
    append_normalized_bytes(
        normalized,
        std::string_view(split.text).substr(pos, end - pos),
        OriginalSpan{split.offset.start + pos, split.offset.start + end});
    pos = end;
  }
  return normalized;
}

NormalizedInput bert_normalize_input(
    const InputSplit & split,
    const detail::BertNormalizerConfig & config) {
  if (!config.enabled) {
    return identity_normalized_input(split);
  }

  NormalizedInput normalized;
  for (std::size_t pos = 0; pos < split.text.size();) {
    const auto end = next_codepoint(split.text, pos);
    const auto original_span =
        OriginalSpan{split.offset.start + pos, split.offset.start + end};
    const auto codepoint = utf8_codepoint_at(split.text, pos);
    const auto lead = static_cast<unsigned char>(split.text[pos]);

    if (config.clean_text) {
      if (is_bert_control_codepoint(codepoint) ||
          (lead < 0x80U && is_bert_control_ascii(lead))) {
        pos = end;
        continue;
      }
      if (is_bert_whitespace_codepoint(codepoint)) {
        append_normalized_bytes(normalized, " ", original_span);
        pos = end;
        continue;
      }
    }

    if (config.handle_chinese_chars && is_chinese_codepoint(codepoint)) {
      append_normalized_bytes(
          normalized,
          " ",
          OriginalSpan{original_span.start, original_span.start});
      append_normalized_bytes(
          normalized,
          std::string_view(split.text).substr(pos, end - pos),
          original_span);
      append_normalized_bytes(
          normalized,
          " ",
          OriginalSpan{original_span.end, original_span.end});
      pos = end;
      continue;
    }

    append_normalized_bytes(
        normalized,
        std::string_view(split.text).substr(pos, end - pos),
        original_span);
    pos = end;
  }

  if (config.strip_accents.value_or(config.lowercase)) {
    normalized = normalize_unicode_normalized(
        normalized,
        unicode::NormalizationForm::Nfd);
    normalized = strip_accents_normalized(normalized);
  }
  if (config.lowercase) {
    normalized = lowercase_unicode_normalized(normalized);
  }
  return normalized;
}

OriginalSpan merged_span(
    const std::vector<OriginalSpan> & spans,
    std::size_t start,
    std::size_t end) {
  if (spans.empty()) {
    return OriginalSpan{0, 0};
  }
  start = std::min(start, spans.size());
  end = std::min(end, spans.size());
  if (start >= end) {
    const auto index = std::min(start, spans.size() - 1);
    const auto value = spans[index];
    return OriginalSpan{value.start, value.start};
  }

  std::size_t original_start = spans[start].start;
  std::size_t original_end = spans[start].end;
  for (std::size_t index = start; index < end; ++index) {
    original_start = std::min(original_start, spans[index].start);
    original_end = std::max(original_end, spans[index].end);
  }
  return OriginalSpan{original_start, original_end};
}

NormalizedInput replace_normalized(
    const NormalizedInput & input,
    const std::string & pattern,
    const std::string & content) {
  if (pattern.empty()) {
    return input;
  }

  NormalizedInput output;
  for (std::size_t pos = 0; pos < input.text.size();) {
    if (pos + pattern.size() <= input.text.size() &&
        input.text.compare(pos, pattern.size(), pattern) == 0) {
      append_normalized_bytes(
          output,
          content,
          merged_span(input.normalized_byte_spans, pos, pos + pattern.size()));
      pos += pattern.size();
      continue;
    }

    const auto end = next_codepoint(input.text, pos);
    append_normalized_bytes(
        output,
        std::string_view(input.text).substr(pos, end - pos),
        merged_span(input.normalized_byte_spans, pos, end));
    pos = end;
  }
  return output;
}

NormalizedInput replace_matches_normalized(
    const NormalizedInput & input,
    const std::vector<unicode::RegexMatch> & matches,
    const std::string & content) {
  NormalizedInput output;
  std::size_t pos = 0;
  for (const auto & match : matches) {
    if (match.start > pos) {
      append_normalized_bytes(
          output,
          std::string_view(input.text).substr(pos, match.start - pos),
          merged_span(input.normalized_byte_spans, pos, match.start));
    }
    if (match.start < match.end) {
      append_normalized_bytes(
          output,
          content,
          merged_span(input.normalized_byte_spans, match.start, match.end));
    }
    pos = std::max(pos, match.end);
  }
  if (pos < input.text.size()) {
    append_normalized_bytes(
        output,
        std::string_view(input.text).substr(pos),
        merged_span(input.normalized_byte_spans, pos, input.text.size()));
  }
  return output;
}

NormalizedInput replace_regex_normalized(
    const NormalizedInput & input,
    const std::string & pattern,
    const std::string & content) {
  return replace_matches_normalized(
      input,
      unicode::find_regex_matches_utf8(input.text, pattern),
      content);
}

NormalizedInput lowercase_unicode_normalized(const NormalizedInput & input) {
  NormalizedInput output;
  const auto lowered = unicode::lowercase_utf8_with_spans(input.text);
  output.text = lowered.text;
  output.normalized_byte_spans.reserve(lowered.byte_spans.size());
  for (const auto & span : lowered.byte_spans) {
    output.normalized_byte_spans.push_back(
        merged_span(input.normalized_byte_spans, span.start, span.end));
  }
  return output;
}

int base64_value(char value) {
  if (value >= 'A' && value <= 'Z') {
    return value - 'A';
  }
  if (value >= 'a' && value <= 'z') {
    return value - 'a' + 26;
  }
  if (value >= '0' && value <= '9') {
    return value - '0' + 52;
  }
  if (value == '+') {
    return 62;
  }
  if (value == '/') {
    return 63;
  }
  return -1;
}

std::vector<unsigned char> decode_base64(std::string_view encoded) {
  std::vector<unsigned char> decoded;
  decoded.reserve((encoded.size() * 3) / 4);

  int accumulator = 0;
  int bits = -8;
  for (const char value : encoded) {
    if (value == '=') {
      break;
    }
    if (std::isspace(static_cast<unsigned char>(value)) != 0) {
      continue;
    }
    const int digit = base64_value(value);
    if (digit < 0) {
      throw std::runtime_error("Precompiled precompiled_charsmap is not base64");
    }
    accumulator = (accumulator << 6) | digit;
    bits += 6;
    if (bits >= 0) {
      decoded.push_back(static_cast<unsigned char>((accumulator >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return decoded;
}

std::uint32_t read_le_u32(
    const std::vector<unsigned char> & bytes,
    std::size_t offset) {
  if (offset + 4 > bytes.size()) {
    throw std::runtime_error("Precompiled charsmap is truncated");
  }
  return static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

detail::PrecompiledCharsMap parse_precompiled_charsmap(std::string_view encoded) {
  const auto bytes = decode_base64(encoded);
  if (bytes.size() < 4) {
    throw std::runtime_error("Precompiled charsmap is too small");
  }

  const auto trie_size = read_le_u32(bytes, 0);
  if (trie_size % 4 != 0) {
    throw std::runtime_error("Precompiled charsmap trie size must be divisible by four");
  }
  if (static_cast<std::size_t>(trie_size) + 4 > bytes.size()) {
    throw std::runtime_error("Precompiled charsmap trie extends past input");
  }

  detail::PrecompiledCharsMap map;
  map.enabled = true;
  map.trie.reserve(trie_size / 4);
  for (std::size_t offset = 4; offset < 4 + trie_size; offset += 4) {
    map.trie.push_back(read_le_u32(bytes, offset));
  }
  map.normalized.assign(
      reinterpret_cast<const char *>(bytes.data() + 4 + trie_size),
      bytes.size() - 4 - trie_size);
  return map;
}

bool precompiled_has_leaf(std::uint32_t unit) {
  return ((unit >> 8U) & 1U) == 1U;
}

std::uint32_t precompiled_value(std::uint32_t unit) {
  return unit & ((1U << 31U) - 1U);
}

std::uint32_t precompiled_label(std::uint32_t unit) {
  return unit & ((1U << 31U) | 0xFFU);
}

std::uint32_t precompiled_offset(std::uint32_t unit) {
  return (unit >> 10U) << ((unit & (1U << 9U)) >> 6U);
}

std::vector<std::uint32_t> precompiled_common_prefix_search(
    const detail::PrecompiledCharsMap & map,
    std::string_view key) {
  std::vector<std::uint32_t> results;
  if (map.trie.empty()) {
    return results;
  }

  std::uint32_t node_pos = 0;
  auto unit = map.trie[node_pos];
  node_pos ^= precompiled_offset(unit);
  for (const unsigned char byte : key) {
    if (byte == 0) {
      break;
    }
    node_pos ^= byte;
    if (node_pos >= map.trie.size()) {
      return results;
    }
    unit = map.trie[node_pos];
    if (precompiled_label(unit) != byte) {
      return results;
    }
    node_pos ^= precompiled_offset(unit);
    if (node_pos >= map.trie.size()) {
      return results;
    }
    if (precompiled_has_leaf(unit)) {
      results.push_back(precompiled_value(map.trie[node_pos]));
    }
  }
  return results;
}

std::optional<std::string_view> precompiled_transform(
    const detail::PrecompiledCharsMap & map,
    std::string_view chunk) {
  const auto results = precompiled_common_prefix_search(map, chunk);
  if (results.empty()) {
    return std::nullopt;
  }

  const auto start = static_cast<std::size_t>(results.front());
  if (start >= map.normalized.size()) {
    return std::nullopt;
  }
  std::size_t end = start;
  while (end < map.normalized.size() && map.normalized[end] != '\0') {
    ++end;
  }
  return std::string_view(map.normalized).substr(start, end - start);
}

bool is_combining_mark_codepoint(std::uint32_t codepoint) {
  return unicode::is_nonspacing_mark(codepoint);
}

std::size_t normalization_chunk_end(
    std::string_view text,
    std::size_t start,
    unicode::NormalizationForm form) {
  auto end = next_codepoint(text, start);
  if (form != unicode::NormalizationForm::Nfc &&
      form != unicode::NormalizationForm::Nfkc) {
    return end;
  }

  while (end < text.size()) {
    const auto codepoint = utf8_codepoint_at(text, end);
    if (!is_combining_mark_codepoint(codepoint)) {
      break;
    }
    end = next_codepoint(text, end);
  }
  return end;
}

NormalizedInput normalize_unicode_normalized(
    const NormalizedInput & input,
    unicode::NormalizationForm form) {
  NormalizedInput output;
  for (std::size_t pos = 0; pos < input.text.size();) {
    const auto end = normalization_chunk_end(input.text, pos, form);
    const auto span = merged_span(input.normalized_byte_spans, pos, end);
    const auto current = std::string_view(input.text).substr(pos, end - pos);
    const auto normalized = unicode::normalize_utf8(current, form);
    append_normalized_bytes(output, normalized.text, span);
    pos = end;
  }
  return output;
}

bool is_strip_whitespace_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && unicode::is_whitespace(utf8_codepoint_at(text, pos));
}

NormalizedInput strip_normalized(
    const NormalizedInput & input,
    bool strip_left,
    bool strip_right) {
  std::size_t start = 0;
  auto end = input.text.size();

  if (strip_left) {
    while (start < end && is_strip_whitespace_at(input.text, start)) {
      start = next_codepoint(input.text, start);
    }
  }

  if (strip_right) {
    while (end > start) {
      const auto previous = previous_codepoint_start(input.text, end);
      if (!is_strip_whitespace_at(input.text, previous)) {
        break;
      }
      end = previous;
    }
  }

  NormalizedInput output;
  for (std::size_t pos = start; pos < end;) {
    const auto current_end = next_codepoint(input.text, pos);
    append_normalized_bytes(
        output,
        std::string_view(input.text).substr(pos, current_end - pos),
        merged_span(input.normalized_byte_spans, pos, current_end));
    pos = current_end;
  }
  return output;
}

bool should_remove_nmt_codepoint(std::uint32_t codepoint) {
  return (codepoint >= 0x0001U && codepoint <= 0x0008U) ||
      codepoint == 0x000BU ||
      (codepoint >= 0x000EU && codepoint <= 0x001FU) ||
      codepoint == 0x007FU ||
      codepoint == 0x008FU ||
      codepoint == 0x009FU;
}

bool should_map_nmt_to_space(std::uint32_t codepoint) {
  return codepoint == 0x0009U ||
      codepoint == 0x000AU ||
      codepoint == 0x000CU ||
      codepoint == 0x000DU ||
      codepoint == 0x1680U ||
      (codepoint >= 0x200BU && codepoint <= 0x200FU) ||
      codepoint == 0x2028U ||
      codepoint == 0x2029U ||
      codepoint == 0x2581U ||
      codepoint == 0xFEFFU ||
      codepoint == 0xFFFDU;
}

NormalizedInput nmt_normalized(const NormalizedInput & input) {
  NormalizedInput output;
  for (std::size_t pos = 0; pos < input.text.size();) {
    const auto end = next_codepoint(input.text, pos);
    const auto codepoint = utf8_codepoint_at(input.text, pos);
    if (!should_remove_nmt_codepoint(codepoint)) {
      append_normalized_bytes(
          output,
          should_map_nmt_to_space(codepoint)
              ? std::string_view(" ")
              : std::string_view(input.text).substr(pos, end - pos),
          merged_span(input.normalized_byte_spans, pos, end));
    }
    pos = end;
  }
  return output;
}

NormalizedInput prepend_normalized(
    const NormalizedInput & input,
    const std::string & prepend) {
  if (input.text.empty() || prepend.empty()) {
    return input;
  }

  NormalizedInput output;
  const auto first_end = next_codepoint(input.text, 0);
  append_normalized_bytes(
      output,
      prepend,
      merged_span(input.normalized_byte_spans, 0, first_end));
  for (std::size_t pos = 0; pos < input.text.size();) {
    const auto end = next_codepoint(input.text, pos);
    append_normalized_bytes(
        output,
        std::string_view(input.text).substr(pos, end - pos),
        merged_span(input.normalized_byte_spans, pos, end));
    pos = end;
  }
  return output;
}

NormalizedInput strip_accents_normalized(const NormalizedInput & input) {
  NormalizedInput output;
  for (std::size_t pos = 0; pos < input.text.size();) {
    const auto end = next_codepoint(input.text, pos);
    const auto codepoint = utf8_codepoint_at(input.text, pos);
    if (!is_combining_mark_codepoint(codepoint)) {
      append_normalized_bytes(
          output,
          std::string_view(input.text).substr(pos, end - pos),
          merged_span(input.normalized_byte_spans, pos, end));
    }
    pos = end;
  }
  return output;
}

std::size_t simple_grapheme_end(std::string_view text, std::size_t start) {
  auto end = next_codepoint(text, start);
  while (end < text.size()) {
    const auto codepoint = utf8_codepoint_at(text, end);
    if (!is_combining_mark_codepoint(codepoint)) {
      break;
    }
    end = next_codepoint(text, end);
  }
  return end;
}

NormalizedInput apply_precompiled_charsmap(
    const NormalizedInput & input,
    const detail::PrecompiledCharsMap & map) {
  if (!map.enabled) {
    return input;
  }

  NormalizedInput output;
  for (std::size_t pos = 0; pos < input.text.size();) {
    const auto grapheme_end = simple_grapheme_end(input.text, pos);
    const auto grapheme = std::string_view(input.text).substr(
        pos,
        grapheme_end - pos);
    if (grapheme.size() < 6) {
      if (const auto transformed = precompiled_transform(map, grapheme)) {
        append_normalized_bytes(
            output,
            *transformed,
            merged_span(input.normalized_byte_spans, pos, grapheme_end));
        pos = grapheme_end;
        continue;
      }
    }

    for (std::size_t current = pos; current < grapheme_end;) {
      const auto current_end = next_codepoint(input.text, current);
      const auto current_text = std::string_view(input.text).substr(
          current,
          current_end - current);
      if (const auto transformed = precompiled_transform(map, current_text)) {
        append_normalized_bytes(
            output,
            *transformed,
            merged_span(input.normalized_byte_spans, current, current_end));
      } else {
        append_normalized_bytes(
            output,
            current_text,
            merged_span(input.normalized_byte_spans, current, current_end));
      }
      current = current_end;
    }
    pos = grapheme_end;
  }
  return output;
}

NormalizedInput simple_normalize_input(
    const InputSplit & split,
    const detail::SimpleNormalizerConfig & config) {
  auto normalized = identity_normalized_input(split);
  if (!config.enabled) {
    return normalized;
  }

  if (!config.ops.empty()) {
    for (const auto & op : config.ops) {
      switch (op.kind) {
        case detail::SimpleNormalizerOpKind::Replace:
          normalized = op.regex_pattern
              ? replace_regex_normalized(normalized, op.pattern, op.content)
              : replace_normalized(normalized, op.pattern, op.content);
          break;
        case detail::SimpleNormalizerOpKind::Nfc:
          normalized = normalize_unicode_normalized(
              normalized,
              unicode::NormalizationForm::Nfc);
          break;
        case detail::SimpleNormalizerOpKind::Nfd:
          normalized = normalize_unicode_normalized(
              normalized,
              unicode::NormalizationForm::Nfd);
          break;
        case detail::SimpleNormalizerOpKind::Nfkc:
          normalized = normalize_unicode_normalized(
              normalized,
              unicode::NormalizationForm::Nfkc);
          break;
        case detail::SimpleNormalizerOpKind::Nfkd:
          normalized = normalize_unicode_normalized(
              normalized,
              unicode::NormalizationForm::Nfkd);
          break;
        case detail::SimpleNormalizerOpKind::Nmt:
          normalized = nmt_normalized(normalized);
          break;
        case detail::SimpleNormalizerOpKind::Prepend:
          normalized = prepend_normalized(normalized, op.content);
          break;
        case detail::SimpleNormalizerOpKind::Strip:
          normalized = strip_normalized(normalized, op.strip_left, op.strip_right);
          break;
        case detail::SimpleNormalizerOpKind::StripAccents:
          normalized = strip_accents_normalized(normalized);
          break;
        case detail::SimpleNormalizerOpKind::Lowercase:
          normalized = lowercase_unicode_normalized(normalized);
          break;
        case detail::SimpleNormalizerOpKind::Precompiled:
          if (op.precompiled_index >= config.precompiled_maps.size()) {
            throw std::runtime_error("Precompiled normalizer index is out of range");
          }
          normalized = apply_precompiled_charsmap(
              normalized,
              config.precompiled_maps[op.precompiled_index]);
          break;
      }
    }
    return normalized;
  }

  for (const auto & replacement : config.replacements) {
    normalized = replace_normalized(normalized, replacement.first, replacement.second);
  }
  if (config.nfkc) {
    normalized = normalize_unicode_normalized(
        normalized,
        unicode::NormalizationForm::Nfkc);
  } else if (config.nfkd) {
    normalized = normalize_unicode_normalized(
        normalized,
        unicode::NormalizationForm::Nfkd);
  } else if (config.nfc) {
    normalized = normalize_unicode_normalized(
        normalized,
        unicode::NormalizationForm::Nfc);
  } else if (config.nfd) {
    normalized = normalize_unicode_normalized(
        normalized,
        unicode::NormalizationForm::Nfd);
  }
  if (config.strip) {
    normalized = strip_normalized(normalized, true, true);
  }
  if (config.nmt) {
    normalized = nmt_normalized(normalized);
  }
  if (config.strip_accents) {
    normalized = strip_accents_normalized(normalized);
  }
  if (config.lowercase) {
    normalized = lowercase_unicode_normalized(normalized);
  }
  return normalized;
}

Offset convert_normalized_range(
    const NormalizedInput & normalized,
    std::size_t start,
    std::size_t end) {
  if (normalized.normalized_byte_spans.empty()) {
    return Offset{0, 0};
  }
  start = std::min(start, normalized.normalized_byte_spans.size());
  end = std::min(end, normalized.normalized_byte_spans.size());
  if (start >= end) {
    const auto index = std::min(start, normalized.normalized_byte_spans.size() - 1);
    const auto span = normalized.normalized_byte_spans[index];
    return Offset{span.start, span.end};
  }

  auto original_start = std::numeric_limits<std::size_t>::max();
  std::size_t original_end = 0;
  for (std::size_t index = start; index < end; ++index) {
    const auto span = normalized.normalized_byte_spans[index];
    original_start = std::min(original_start, span.start);
    original_end = std::max(original_end, span.end);
  }
  return Offset{original_start, original_end};
}

std::size_t consume_ascii_letters(std::string_view text, std::size_t pos) {
  while (is_ascii_letter_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_ascii_numbers(std::string_view text, std::size_t pos) {
  while (is_ascii_number_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_other(std::string_view text, std::size_t pos) {
  while (is_other_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_ascii_spaces(std::string_view text, std::size_t pos) {
  while (is_ascii_space_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_llama_letters(std::string_view text, std::size_t pos) {
  while (is_llama_letter_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_llama_numbers(std::string_view text, std::size_t pos) {
  std::size_t count = 0;
  while (count < 3 && is_llama_number_at(text, pos)) {
    pos = next_codepoint(text, pos);
    ++count;
  }
  return pos;
}

std::size_t consume_llama_other(std::string_view text, std::size_t pos) {
  while (is_llama_other_at(text, pos) && !is_llama_line_break_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  while (is_llama_line_break_at(text, pos)) {
    pos = next_codepoint(text, pos);
  }
  return pos;
}

std::size_t consume_llama_spaces(std::string_view text, std::size_t pos) {
  std::size_t scan = pos;
  bool saw_line_break = false;
  while (is_llama_space_at(text, scan)) {
    if (is_llama_line_break_at(text, scan)) {
      saw_line_break = true;
    } else if (saw_line_break) {
      break;
    }
    scan = next_codepoint(text, scan);
  }
  if (saw_line_break) {
    return scan;
  }

  std::size_t count = 0;
  scan = pos;
  while (is_llama_space_at(text, scan) && !is_llama_line_break_at(text, scan)) {
    scan = next_codepoint(text, scan);
    ++count;
  }
  if (count > 1 && scan < text.size()) {
    std::size_t end = pos;
    for (std::size_t index = 0; index + 1 < count; ++index) {
      end = next_codepoint(text, end);
    }
    return end;
  }
  return scan;
}

bool ascii_iequals_at(std::string_view text, std::size_t pos, std::string_view pattern) {
  if (pos + pattern.size() > text.size()) {
    return false;
  }
  for (std::size_t index = 0; index < pattern.size(); ++index) {
    const auto lhs = static_cast<unsigned char>(text[pos + index]);
    const auto rhs = static_cast<unsigned char>(pattern[index]);
    if (std::tolower(lhs) != std::tolower(rhs)) {
      return false;
    }
  }
  return true;
}

std::optional<std::size_t> llama_contraction_end(std::string_view text, std::size_t pos) {
  static const std::string_view contractions[] = {
      "'s",
      "'t",
      "'re",
      "'ve",
      "'m",
      "'ll",
      "'d",
  };
  for (const auto contraction : contractions) {
    if (ascii_iequals_at(text, pos, contraction)) {
      return pos + contraction.size();
    }
  }
  return std::nullopt;
}

std::vector<RawPiece> split_llama_regex(std::string_view text) {
  std::vector<RawPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto start = pos;

    if (const auto contraction_end = llama_contraction_end(text, pos)) {
      pos = *contraction_end;
    } else {
      const auto current_end = next_codepoint(text, pos);
      const auto current = utf8_codepoint_at(text, pos);
      const bool current_can_prefix_letters =
          !is_line_break_codepoint(current) &&
          !is_llama_letter_codepoint(current) &&
          !is_llama_number_codepoint(current);
      if (current_can_prefix_letters && is_llama_letter_at(text, current_end)) {
        pos = consume_llama_letters(text, current_end);
      } else if (is_llama_letter_at(text, pos)) {
        pos = consume_llama_letters(text, pos);
      } else if (is_llama_number_at(text, pos)) {
        pos = consume_llama_numbers(text, pos);
      } else if (current == ' ' && is_llama_other_at(text, current_end) &&
          !is_llama_line_break_at(text, current_end)) {
        pos = consume_llama_other(text, current_end);
      } else if (is_llama_other_at(text, pos) && !is_llama_line_break_at(text, pos)) {
        pos = consume_llama_other(text, pos);
      } else if (is_llama_space_at(text, pos)) {
        pos = consume_llama_spaces(text, pos);
      } else {
        pos = current_end;
      }
    }

    if (start < pos) {
      pieces.push_back(RawPiece{start, pos});
    }
  }
  return pieces;
}

std::vector<RawPiece> split_byte_level_regex(std::string_view text) {
  std::vector<RawPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto start = pos;
    if (is_ascii_space_at(text, pos)) {
      const auto after_space = next_codepoint(text, pos);
      if (is_ascii_letter_at(text, after_space)) {
        pos = consume_ascii_letters(text, after_space);
      } else if (is_ascii_number_at(text, after_space)) {
        pos = consume_ascii_numbers(text, after_space);
      } else if (is_other_at(text, after_space)) {
        pos = consume_other(text, after_space);
      } else {
        pos = consume_ascii_spaces(text, pos);
      }
    } else if (is_ascii_letter_at(text, pos)) {
      pos = consume_ascii_letters(text, pos);
    } else if (is_ascii_number_at(text, pos)) {
      pos = consume_ascii_numbers(text, pos);
    } else {
      pos = consume_other(text, pos);
    }

    if (start < pos) {
      pieces.push_back(RawPiece{start, pos});
    }
  }
  return pieces;
}

bool is_bert_punctuation_codepoint(std::uint32_t codepoint) {
  return unicode::is_punctuation(codepoint);
}

bool is_bert_whitespace_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return false;
  }
  return is_bert_whitespace_codepoint(utf8_codepoint_at(text, pos));
}

bool is_bert_punctuation_at(std::string_view text, std::size_t pos) {
  if (pos >= text.size()) {
    return false;
  }
  return is_bert_punctuation_codepoint(utf8_codepoint_at(text, pos));
}

std::vector<PreTokenPiece> bert_pre_tokenize(std::string_view text) {
  std::vector<PreTokenPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (is_bert_whitespace_at(text, pos)) {
      pos = next_codepoint(text, pos);
    }
    if (pos >= text.size()) {
      break;
    }

    const auto start = pos;
    if (is_bert_punctuation_at(text, pos)) {
      pos = next_codepoint(text, pos);
      pieces.push_back(PreTokenPiece{start, pos});
      continue;
    }

    while (pos < text.size() && !is_bert_whitespace_at(text, pos) &&
        !is_bert_punctuation_at(text, pos)) {
      pos = next_codepoint(text, pos);
    }
    pieces.push_back(PreTokenPiece{start, pos});
  }
  return pieces;
}

std::vector<PreTokenPiece> whitespace_pre_tokenize(std::string_view text) {
  std::vector<PreTokenPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (is_ascii_space_at(text, pos)) {
      pos = next_codepoint(text, pos);
    }
    const auto start = pos;
    while (pos < text.size() && !is_ascii_space_at(text, pos)) {
      pos = next_codepoint(text, pos);
    }
    if (start < pos) {
      pieces.push_back(PreTokenPiece{start, pos});
    }
  }
  return pieces;
}

std::vector<PreTokenPiece> whitespace_regex_pre_tokenize(std::string_view text) {
  std::vector<PreTokenPiece> pieces;
  for (const auto & match : unicode::find_regex_matches_utf8(
           text,
           R"(\w+|[^\w\s]+)")) {
    pieces.push_back(PreTokenPiece{match.start, match.end});
  }
  return pieces;
}

bool is_digits_pre_tokenizer_digit_at(std::string_view text, std::size_t pos) {
  return pos < text.size() && unicode::is_number(utf8_codepoint_at(text, pos));
}

std::vector<RawPiece> digits_pre_tokenize(
    std::string_view text,
    bool individual_digits) {
  std::vector<RawPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto start = pos;
    const auto is_digit = is_digits_pre_tokenizer_digit_at(text, pos);
    if (is_digit && individual_digits) {
      pos = next_codepoint(text, pos);
    } else {
      while (pos < text.size() &&
          is_digits_pre_tokenizer_digit_at(text, pos) == is_digit) {
        pos = next_codepoint(text, pos);
      }
    }
    if (start < pos) {
      pieces.push_back(RawPiece{start, pos});
    }
  }
  return pieces;
}

void append_piece_bytes(
    NormalizedPiece & piece,
    std::string_view bytes,
    OriginalSpan original_span) {
  piece.text.append(bytes);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    piece.normalized_byte_spans.push_back(original_span);
  }
}

NormalizedPiece slice_normalized_piece(
    const NormalizedInput & normalized,
    std::size_t start,
    std::size_t end) {
  NormalizedPiece piece;
  start = std::min(start, normalized.text.size());
  end = std::min(end, normalized.text.size());
  piece.text = normalized.text.substr(start, end - start);
  piece.normalized_byte_spans.insert(
      piece.normalized_byte_spans.end(),
      normalized.normalized_byte_spans.begin() + static_cast<std::ptrdiff_t>(start),
      normalized.normalized_byte_spans.begin() + static_cast<std::ptrdiff_t>(end));
  return piece;
}

std::vector<NormalizedPiece> whitespace_split_normalized(
    const NormalizedInput & normalized) {
  std::vector<NormalizedPiece> pieces;
  for (const auto & piece : whitespace_pre_tokenize(normalized.text)) {
    pieces.push_back(slice_normalized_piece(normalized, piece.start, piece.end));
  }
  return pieces;
}

std::vector<NormalizedPiece> metaspace_split_piece(
    const NormalizedPiece & piece,
    const detail::MetaspaceConfig & config,
    bool is_first_piece) {
  NormalizedPiece replaced;
  for (std::size_t pos = 0; pos < piece.text.size();) {
    const auto end = next_codepoint(piece.text, pos);
    const auto span = merged_span(piece.normalized_byte_spans, pos, end);
    const auto current = std::string_view(piece.text).substr(pos, end - pos);
    if (current.size() == 1 && std::isspace(static_cast<unsigned char>(current[0])) != 0) {
      append_piece_bytes(replaced, config.replacement, span);
    } else {
      append_piece_bytes(replaced, current, span);
    }
    pos = end;
  }

  const bool should_prepend = config.prepend_scheme == "always" ||
      (config.prepend_scheme == "first" && is_first_piece);
  if (should_prepend && !starts_with(replaced.text, config.replacement)) {
    NormalizedPiece prefixed;
    const auto anchor = replaced.normalized_byte_spans.empty()
        ? OriginalSpan{0, 0}
        : replaced.normalized_byte_spans.front();
    append_piece_bytes(prefixed, config.replacement, anchor);
    prefixed.text.append(replaced.text);
    prefixed.normalized_byte_spans.insert(
        prefixed.normalized_byte_spans.end(),
        replaced.normalized_byte_spans.begin(),
        replaced.normalized_byte_spans.end());
    replaced = std::move(prefixed);
  }

  if (!config.split || config.replacement.empty()) {
    return {replaced};
  }

  std::vector<NormalizedPiece> splits;
  std::size_t start = 0;
  while (start < replaced.text.size()) {
    std::size_t next = replaced.text.find(config.replacement, start + 1);
    if (next == std::string::npos) {
      next = replaced.text.size();
    }
    splits.push_back(slice_normalized_piece(
        NormalizedInput{replaced.text, replaced.normalized_byte_spans},
        start,
        next));
    start = next;
  }
  return splits;
}

std::vector<NormalizedPiece> sentencepiece_pre_tokenize(
    const NormalizedInput & normalized,
    bool whitespace_split,
    const detail::MetaspaceConfig & metaspace) {
  std::vector<NormalizedPiece> pieces = whitespace_split
      ? whitespace_split_normalized(normalized)
      : std::vector<NormalizedPiece>{NormalizedPiece{
            normalized.text,
            normalized.normalized_byte_spans}};

  if (!metaspace.enabled) {
    return pieces;
  }

  std::vector<NormalizedPiece> output;
  for (std::size_t index = 0; index < pieces.size(); ++index) {
    auto current = metaspace_split_piece(pieces[index], metaspace, index == 0);
    output.insert(output.end(), current.begin(), current.end());
  }
  return output;
}

Offset convert_piece_offsets(
    const NormalizedPiece & piece,
    std::size_t start,
    std::size_t end) {
  return convert_normalized_range(
      NormalizedInput{piece.text, piece.normalized_byte_spans},
      start,
      end);
}

void append_byte_level_char(
    ByteLevelPiece & piece,
    unsigned char byte,
    OriginalSpan original_span) {
  const auto & mapped = byte_to_unicode()[byte];
  piece.text.append(mapped);
  for (std::size_t i = 0; i < mapped.size(); ++i) {
    piece.normalized_byte_spans.push_back(original_span);
  }
}

ByteLevelPiece byte_level_piece_from_raw(
    std::string_view scan_text,
    const RawPiece & raw,
    bool has_virtual_prefix,
    std::size_t original_base) {
  ByteLevelPiece piece;
  std::size_t pos = raw.start;
  while (pos < raw.end) {
    if (has_virtual_prefix && pos == 0) {
      append_byte_level_char(
          piece,
          static_cast<unsigned char>(' '),
          OriginalSpan{original_base, original_base});
      ++pos;
      continue;
    }

    const auto original_local = has_virtual_prefix ? pos - 1 : pos;
    const auto codepoint_end = next_codepoint(scan_text, pos);
    const auto original_span = OriginalSpan{
        original_base + original_local,
        original_base + original_local + (codepoint_end - pos)};
    for (std::size_t byte_pos = pos; byte_pos < codepoint_end; ++byte_pos) {
      append_byte_level_char(
          piece,
          static_cast<unsigned char>(scan_text[byte_pos]),
          original_span);
    }
    pos = codepoint_end;
  }
  return piece;
}

ByteLevelPiece byte_level_piece_from_normalized_raw(
    std::string_view scan_text,
    const std::vector<OriginalSpan> & scan_byte_spans,
    const RawPiece & raw) {
  ByteLevelPiece piece;
  std::size_t pos = raw.start;
  while (pos < raw.end) {
    const auto codepoint_end = next_codepoint(scan_text, pos);
    for (std::size_t byte_pos = pos; byte_pos < codepoint_end; ++byte_pos) {
      const auto original_span = byte_pos < scan_byte_spans.size()
          ? scan_byte_spans[byte_pos]
          : OriginalSpan{0, 0};
      append_byte_level_char(
          piece,
          static_cast<unsigned char>(scan_text[byte_pos]),
          original_span);
    }
    pos = codepoint_end;
  }
  return piece;
}

std::vector<ByteLevelPiece> byte_level_pre_tokenize(
    const InputSplit & split,
    const detail::ByteLevelConfig & config) {
  const bool has_virtual_prefix =
      config.add_prefix_space && !starts_with(split.text, " ");
  const auto scan_text = has_virtual_prefix ? " " + split.text : split.text;
  const auto raw_pieces = config.use_regex
      ? split_byte_level_regex(scan_text)
      : std::vector<RawPiece>{RawPiece{0, scan_text.size()}};

  std::vector<ByteLevelPiece> pieces;
  pieces.reserve(raw_pieces.size());
  for (const auto & raw : raw_pieces) {
    pieces.push_back(byte_level_piece_from_raw(
        scan_text,
        raw,
        has_virtual_prefix,
        split.offset.start));
  }
  return pieces;
}

std::vector<ByteLevelPiece> byte_level_pre_tokenize(
    const NormalizedPiece & split,
    const detail::ByteLevelConfig & config) {
  const bool has_virtual_prefix =
      config.add_prefix_space && !starts_with(split.text, " ");
  std::string scan_text;
  std::vector<OriginalSpan> scan_byte_spans;
  scan_text.reserve(split.text.size() + (has_virtual_prefix ? 1 : 0));
  scan_byte_spans.reserve(
      split.normalized_byte_spans.size() + (has_virtual_prefix ? 1 : 0));

  if (has_virtual_prefix) {
    scan_text.push_back(' ');
    scan_byte_spans.push_back(split.normalized_byte_spans.empty()
        ? OriginalSpan{0, 0}
        : split.normalized_byte_spans.front());
  }
  scan_text.append(split.text);
  scan_byte_spans.insert(
      scan_byte_spans.end(),
      split.normalized_byte_spans.begin(),
      split.normalized_byte_spans.end());

  const auto raw_pieces = config.use_regex
      ? split_byte_level_regex(scan_text)
      : std::vector<RawPiece>{RawPiece{0, scan_text.size()}};

  std::vector<ByteLevelPiece> pieces;
  pieces.reserve(raw_pieces.size());
  for (const auto & raw : raw_pieces) {
    pieces.push_back(byte_level_piece_from_normalized_raw(
        scan_text,
        scan_byte_spans,
        raw));
  }
  return pieces;
}

ByteLevelPiece byte_level_normalize_split(const InputSplit & split) {
  return byte_level_piece_from_raw(
      split.text,
      RawPiece{0, split.text.size()},
      false,
      split.offset.start);
}

ByteLevelPiece slice_byte_level_piece(
    const ByteLevelPiece & piece,
    std::size_t start,
    std::size_t end) {
  ByteLevelPiece output;
  output.text = piece.text.substr(start, end - start);
  output.normalized_byte_spans.insert(
      output.normalized_byte_spans.end(),
      piece.normalized_byte_spans.begin() + static_cast<std::ptrdiff_t>(start),
      piece.normalized_byte_spans.begin() + static_cast<std::ptrdiff_t>(end));
  return output;
}

bool is_llama_stream_escaped_split_boundary(std::uint32_t codepoint) {
  return codepoint == '\\' ||
      codepoint == 's' ||
      codepoint == 'p' ||
      codepoint == 'L' ||
      codepoint == 'N' ||
      codepoint == '{' ||
      codepoint == '}';
}

std::vector<RawPiece> split_llama_stream_escaped_regex(std::string_view text) {
  std::vector<RawPiece> pieces;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const auto start = pos;
    const auto codepoint = utf8_codepoint_at(text, pos);
    if (is_llama_stream_escaped_split_boundary(codepoint)) {
      pos = next_codepoint(text, pos);
    } else {
      while (pos < text.size() &&
          !is_llama_stream_escaped_split_boundary(utf8_codepoint_at(text, pos))) {
        pos = next_codepoint(text, pos);
      }
    }
    if (start < pos) {
      pieces.push_back(RawPiece{start, pos});
    }
  }
  return pieces;
}

std::vector<SplitSpan> split_spans_from_matches(
    std::string_view text,
    const std::vector<unicode::RegexMatch> & matches,
    bool invert) {
  std::vector<SplitSpan> spans;
  if (text.empty()) {
    return {SplitSpan{0, 0, invert}};
  }

  std::size_t prev = 0;
  for (const auto & match : matches) {
    if (match.start > prev) {
      spans.push_back(SplitSpan{prev, match.start, invert});
    }
    if (match.start < match.end) {
      spans.push_back(SplitSpan{match.start, match.end, !invert});
    }
    prev = std::max(prev, match.end);
  }
  if (prev < text.size()) {
    spans.push_back(SplitSpan{prev, text.size(), invert});
  }
  if (spans.empty()) {
    spans.push_back(SplitSpan{0, text.size(), invert});
  }
  return spans;
}

std::vector<unicode::RegexMatch> find_literal_matches(
    std::string_view text,
    std::string_view pattern) {
  std::vector<unicode::RegexMatch> matches;
  if (pattern.empty()) {
    return matches;
  }

  std::size_t pos = 0;
  while ((pos = text.find(pattern, pos)) != std::string_view::npos) {
    matches.push_back(unicode::RegexMatch{pos, pos + pattern.size()});
    pos += pattern.size();
  }
  return matches;
}

std::vector<SplitSpan> split_pattern_spans(
    std::string_view text,
    const detail::SplitPreTokenizerConfig & config) {
  const auto matches = config.regex_pattern
      ? unicode::find_regex_matches_utf8(text, config.pattern)
      : find_literal_matches(text, config.pattern);
  return split_spans_from_matches(text, matches, config.invert);
}

std::vector<RawPiece> apply_split_behavior(
    const std::vector<SplitSpan> & spans,
    const std::string & behavior) {
  std::vector<SplitSpan> merged;

  if (behavior == "Isolated") {
    merged = spans;
  } else if (behavior == "Removed") {
    for (const auto & span : spans) {
      if (!span.is_match) {
        merged.push_back(span);
      }
    }
  } else if (behavior == "Contiguous") {
    bool previous_match = false;
    for (const auto & span : spans) {
      if (!merged.empty() && span.is_match == previous_match) {
        merged.back().end = span.end;
      } else {
        merged.push_back(span);
      }
      previous_match = span.is_match;
    }
  } else if (behavior == "MergedWithPrevious") {
    bool previous_match = false;
    for (const auto & span : spans) {
      if (span.is_match && !previous_match && !merged.empty()) {
        merged.back().end = span.end;
      } else {
        merged.push_back(span);
      }
      previous_match = span.is_match;
    }
  } else if (behavior == "MergedWithNext") {
    bool previous_match = false;
    for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
      const auto & span = *it;
      if (span.is_match && !previous_match && !merged.empty()) {
        merged.back().start = span.start;
      } else {
        merged.push_back(span);
      }
      previous_match = span.is_match;
    }
    std::reverse(merged.begin(), merged.end());
  } else {
    throw std::runtime_error("unsupported Split pre_tokenizer behavior");
  }

  std::vector<RawPiece> pieces;
  pieces.reserve(merged.size());
  for (const auto & span : merged) {
    if (span.start < span.end) {
      pieces.push_back(RawPiece{span.start, span.end});
    }
  }
  return pieces;
}

std::vector<RawPiece> split_generic_pattern(
    std::string_view text,
    const detail::SplitPreTokenizerConfig & config) {
  return apply_split_behavior(split_pattern_spans(text, config), config.behavior);
}

std::vector<InputSplit> split_pre_tokenize(
    const InputSplit & split,
    const detail::SplitPreTokenizerConfig & config) {
  if (!config.enabled) {
    return {split};
  }

  std::vector<InputSplit> output;
  const auto raw_pieces = config.llama_stream_escaped_regex
      ? split_llama_stream_escaped_regex(split.text)
      : config.llama_regex
          ? split_llama_regex(split.text)
          : split_generic_pattern(split.text, config);
  for (const auto & raw : raw_pieces) {
    output.push_back(InputSplit{
        false,
        0,
        split.text.substr(raw.start, raw.end - raw.start),
        Offset{split.offset.start + raw.start, split.offset.start + raw.end}});
  }
  return output;
}

std::vector<RawPiece> raw_pieces_for_pre_tokenizer_step(
    std::string_view text,
    const detail::PreTokenizerStepConfig & step) {
  if (step.kind == detail::PreTokenizerStepKind::Whitespace) {
    std::vector<RawPiece> pieces;
    for (const auto & piece : whitespace_regex_pre_tokenize(text)) {
      pieces.push_back(RawPiece{piece.start, piece.end});
    }
    return pieces;
  }
  if (step.kind == detail::PreTokenizerStepKind::Digits) {
    return digits_pre_tokenize(text, step.individual_digits);
  }

  if (step.split.llama_stream_escaped_regex) {
    return split_llama_stream_escaped_regex(text);
  }
  if (step.split.llama_regex) {
    return split_llama_regex(text);
  }
  return split_generic_pattern(text, step.split);
}

std::vector<InputSplit> apply_pre_tokenizer_steps(
    const InputSplit & split,
    const std::vector<detail::PreTokenizerStepConfig> & steps) {
  if (steps.empty()) {
    return {split};
  }

  std::vector<InputSplit> pieces{split};
  for (const auto & step : steps) {
    std::vector<InputSplit> next;
    for (const auto & piece : pieces) {
      for (const auto & raw : raw_pieces_for_pre_tokenizer_step(piece.text, step)) {
        next.push_back(InputSplit{
            false,
            0,
            piece.text.substr(raw.start, raw.end - raw.start),
            Offset{piece.offset.start + raw.start, piece.offset.start + raw.end}});
      }
    }
    pieces = std::move(next);
  }
  return pieces;
}

std::vector<NormalizedPiece> split_normalized_piece(
    const NormalizedPiece & split,
    const detail::SplitPreTokenizerConfig & config) {
  if (!config.enabled) {
    return {split};
  }

  const auto raw_pieces = config.llama_stream_escaped_regex
      ? split_llama_stream_escaped_regex(split.text)
      : config.llama_regex
          ? split_llama_regex(split.text)
          : split_generic_pattern(split.text, config);
  const auto normalized = NormalizedInput{
      split.text,
      split.normalized_byte_spans};
  std::vector<NormalizedPiece> output;
  output.reserve(raw_pieces.size());
  for (const auto & raw : raw_pieces) {
    output.push_back(slice_normalized_piece(normalized, raw.start, raw.end));
  }
  return output;
}

std::vector<NormalizedPiece> apply_pre_tokenizer_steps(
    const NormalizedPiece & split,
    const std::vector<detail::PreTokenizerStepConfig> & steps) {
  if (steps.empty()) {
    return {split};
  }

  std::vector<NormalizedPiece> pieces{split};
  for (const auto & step : steps) {
    std::vector<NormalizedPiece> next;
    for (const auto & piece : pieces) {
      const auto normalized = NormalizedInput{
          piece.text,
          piece.normalized_byte_spans};
      for (const auto & raw : raw_pieces_for_pre_tokenizer_step(piece.text, step)) {
        next.push_back(slice_normalized_piece(normalized, raw.start, raw.end));
      }
    }
    pieces = std::move(next);
  }
  return pieces;
}

std::vector<ByteLevelPiece> split_byte_level_normalized_piece(
    const ByteLevelPiece & piece,
    const detail::SplitPreTokenizerConfig & config) {
  if (!config.enabled) {
    return {piece};
  }

  const auto raw_pieces = config.llama_stream_escaped_regex
      ? split_llama_stream_escaped_regex(piece.text)
      : config.llama_regex
          ? split_llama_regex(piece.text)
          : split_generic_pattern(piece.text, config);
  std::vector<ByteLevelPiece> output;
  output.reserve(raw_pieces.size());
  for (const auto & raw : raw_pieces) {
    output.push_back(slice_byte_level_piece(piece, raw.start, raw.end));
  }
  return output;
}

Offset convert_normalized_offsets(
    const ByteLevelPiece & piece,
    std::size_t start,
    std::size_t end) {
  if (piece.normalized_byte_spans.empty()) {
    return Offset{0, 0};
  }
  start = std::min(start, piece.normalized_byte_spans.size());
  end = std::min(end, piece.normalized_byte_spans.size());
  if (start >= end) {
    const auto index = std::min(start, piece.normalized_byte_spans.size() - 1);
    const auto span = piece.normalized_byte_spans[index];
    return Offset{span.start, span.end};
  }

  auto original_start = std::numeric_limits<std::size_t>::max();
  std::size_t original_end = 0;
  for (std::size_t index = start; index < end; ++index) {
    const auto span = piece.normalized_byte_spans[index];
    original_start = std::min(original_start, span.start);
    original_end = std::max(original_end, span.end);
  }
  return Offset{original_start, original_end};
}

std::uint32_t require_token_id(
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const std::string & token);

std::uint32_t cached_or_required_token_id(
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const std::string & token,
    const std::optional<std::uint32_t> & cached_id);

std::vector<BpeToken> bpe_symbols_to_tokens(
    const ByteLevelPiece & piece,
    const std::vector<std::string> & id_to_token,
    const std::vector<BpeSymbol> & symbols) {
  std::vector<BpeToken> tokens;
  tokens.reserve(symbols.size());
  for (const auto & symbol : symbols) {
    if (symbol.id >= id_to_token.size()) {
      continue;
    }
    tokens.push_back(BpeToken{
        symbol.id,
        id_to_token[symbol.id],
        convert_normalized_offsets(piece, symbol.start, symbol.end)});
  }
  return tokens;
}

std::vector<BpeSymbol> merge_bpe_symbols(
    std::vector<BpeSymbol> symbols,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config);

std::vector<BpeSymbol> tokenize_bpe_symbols(
    std::string_view piece,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config) {
  std::vector<BpeSymbol> symbols;

  if (config.ignore_merges) {
    const auto found = token_to_id.find(std::string(piece));
    if (found != token_to_id.end()) {
      return {BpeSymbol{found->second, 0, piece.size()}};
    }
  }

  std::optional<BpeSymbol> pending_unk;
  for (std::size_t pos = 0; pos < piece.size();) {
    const auto end = next_codepoint(piece, pos);
    const auto is_first = pos == 0;
    const auto is_last = end == piece.size();
    auto symbol = std::string(piece.substr(pos, end - pos));

    if (!is_first && config.continuing_subword_prefix) {
      symbol = *config.continuing_subword_prefix + symbol;
    }
    if (is_last && config.end_of_word_suffix) {
      symbol += *config.end_of_word_suffix;
    }

    const auto found = token_to_id.find(symbol);
    if (found != token_to_id.end()) {
      flush_pending_unk(symbols, pending_unk);
      symbols.push_back(BpeSymbol{found->second, pos, end});
      pos = end;
      continue;
    }

    if (config.byte_fallback) {
      std::vector<BpeSymbol> fallback_symbols;
      fallback_symbols.reserve(symbol.size());
      bool found_all_fallback_bytes = true;
      for (std::size_t byte_index = 0; byte_index < symbol.size(); ++byte_index) {
        const auto byte = static_cast<unsigned char>(symbol[byte_index]);
        std::optional<std::uint32_t> byte_id = config.byte_fallback_ids[byte];
        if (!byte_id) {
          const auto token = bpe_byte_fallback_token(byte);
          const auto byte_found = token_to_id.find(token);
          if (byte_found == token_to_id.end()) {
            found_all_fallback_bytes = false;
            break;
          }
          byte_id = byte_found->second;
        }
        const auto byte_start =
            symbol.size() == end - pos ? pos + byte_index : pos;
        const auto byte_end =
            symbol.size() == end - pos ? byte_start + 1 : end;
        fallback_symbols.push_back(BpeSymbol{*byte_id, byte_start, byte_end});
      }
      if (found_all_fallback_bytes) {
        flush_pending_unk(symbols, pending_unk);
        symbols.insert(symbols.end(), fallback_symbols.begin(), fallback_symbols.end());
        pos = end;
        continue;
      }
    }

    if (config.unk_token) {
      const auto unk_id =
          cached_or_required_token_id(token_to_id, *config.unk_token, config.unk_id);
      if (pending_unk && config.fuse_unk) {
        pending_unk->end = end;
      } else {
        flush_pending_unk(symbols, pending_unk);
        pending_unk = BpeSymbol{unk_id, pos, end};
      }
    }
    pos = end;
  }
  flush_pending_unk(symbols, pending_unk);

  return merge_bpe_symbols(std::move(symbols), merges, config);
}

std::vector<BpeSymbol> merge_bpe_symbols_linear(
    std::vector<BpeSymbol> symbols,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config) {
  while (symbols.size() > 1) {
    std::optional<std::size_t> best_index;
    detail::BpeMerge best_merge;
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
      const auto found = merges.find(bpe_pair_key(symbols[index].id, symbols[index + 1].id));
      if (found == merges.end()) {
        continue;
      }
      if (should_skip_bpe_merge(config.dropout)) {
        continue;
      }
      if (!best_index || found->second.rank < best_merge.rank) {
        best_index = index;
        best_merge = found->second;
      }
    }
    if (!best_index) {
      break;
    }

    auto & left = symbols[*best_index];
    const auto & right = symbols[*best_index + 1];
    left.id = best_merge.new_id;
    left.end = right.end;
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(*best_index + 1));
  }

  return symbols;
}

std::vector<BpeSymbol> merge_bpe_symbols_heap(
    std::vector<BpeSymbol> symbols,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges) {
  if (symbols.size() <= 1 || merges.empty()) {
    return symbols;
  }

  std::vector<BpeMergeNode> nodes;
  nodes.reserve(symbols.size());
  for (std::size_t index = 0; index < symbols.size(); ++index) {
    nodes.push_back(BpeMergeNode{
        symbols[index],
        index > 0 ? std::optional<std::size_t>(index - 1) : std::nullopt,
        index + 1 < symbols.size() ? std::optional<std::size_t>(index + 1) : std::nullopt,
        0,
        true});
  }

  std::priority_queue<
      BpeMergeCandidate,
      std::vector<BpeMergeCandidate>,
      BpeMergeCandidateCompare>
      candidates;

  const auto push_candidate = [&nodes, &merges, &candidates](std::size_t left_index) {
    if (left_index >= nodes.size() || !nodes[left_index].active ||
        !nodes[left_index].next) {
      return;
    }
    const auto right_index = *nodes[left_index].next;
    if (right_index >= nodes.size() || !nodes[right_index].active) {
      return;
    }
    const auto found = merges.find(
        bpe_pair_key(nodes[left_index].symbol.id, nodes[right_index].symbol.id));
    if (found == merges.end()) {
      return;
    }
    candidates.push(BpeMergeCandidate{
        found->second.rank,
        left_index,
        nodes[left_index].generation,
        right_index,
        nodes[right_index].generation,
        found->second});
  };

  for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
    push_candidate(index);
  }

  std::size_t active_count = nodes.size();
  while (active_count > 1 && !candidates.empty()) {
    const auto candidate = candidates.top();
    candidates.pop();
    if (candidate.left >= nodes.size() || candidate.right >= nodes.size()) {
      continue;
    }

    auto & left = nodes[candidate.left];
    auto & right = nodes[candidate.right];
    if (!left.active || !right.active || left.next != candidate.right ||
        right.previous != candidate.left ||
        left.generation != candidate.left_generation ||
        right.generation != candidate.right_generation) {
      continue;
    }

    left.symbol.id = candidate.merge.new_id;
    left.symbol.end = right.symbol.end;
    ++left.generation;
    right.active = false;
    ++right.generation;

    const auto previous = left.previous;
    const auto next = right.next;
    left.next = next;
    if (next) {
      nodes[*next].previous = candidate.left;
    }
    --active_count;

    if (previous) {
      push_candidate(*previous);
    }
    push_candidate(candidate.left);
  }

  std::vector<BpeSymbol> output;
  output.reserve(active_count);
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (nodes[index].active) {
      output.push_back(nodes[index].symbol);
    }
  }
  return output;
}

std::vector<BpeSymbol> merge_bpe_symbols(
    std::vector<BpeSymbol> symbols,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config) {
  if (config.dropout && *config.dropout > 0.0) {
    return merge_bpe_symbols_linear(std::move(symbols), merges, config);
  }
  return merge_bpe_symbols_heap(std::move(symbols), merges);
}

bool can_cache_bpe_piece(
    std::uint64_t cache_id,
    std::string_view piece,
    const detail::BpeConfig & config) {
  return cache_id != 0 && piece.size() < kBpeCacheMaxLength &&
      (!config.dropout || *config.dropout <= 0.0);
}

std::vector<BpeSymbol> tokenize_bpe_symbols_with_cache(
    std::string_view piece,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config,
    std::uint64_t cache_id) {
  if (!can_cache_bpe_piece(cache_id, piece, config)) {
    return tokenize_bpe_symbols(piece, token_to_id, merges, config);
  }

  auto & cache = bpe_thread_cache[cache_id];
  const auto key = std::string(piece);
  const auto hit = cache.find(key);
  if (hit != cache.end()) {
    return hit->second;
  }

  auto symbols = tokenize_bpe_symbols(piece, token_to_id, merges, config);
  if (cache.size() < kBpeCacheCapacity) {
    cache.emplace(key, symbols);
  }
  return symbols;
}

std::vector<BpeToken> tokenize_bpe_piece(
    const ByteLevelPiece & piece,
    const std::vector<std::string> & id_to_token,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const HashMap<std::uint64_t, detail::BpeMerge> & merges,
    const detail::BpeConfig & config,
    std::uint64_t cache_id) {
  return bpe_symbols_to_tokens(
      piece,
      id_to_token,
      tokenize_bpe_symbols_with_cache(
          piece.text,
          token_to_id,
          merges,
          config,
          cache_id));
}

std::size_t count_codepoints(std::string_view text) {
  std::size_t count = 0;
  for (std::size_t pos = 0; pos < text.size(); pos = next_codepoint(text, pos)) {
    ++count;
  }
  return count;
}

std::uint32_t require_token_id(
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const std::string & token) {
  const auto found = token_to_id.find(token);
  if (found == token_to_id.end()) {
    throw std::runtime_error("required token missing from vocabulary: " + token);
  }
  return found->second;
}

std::uint32_t cached_or_required_token_id(
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const std::string & token,
    const std::optional<std::uint32_t> & cached_id) {
  if (cached_id) {
    return *cached_id;
  }
  return require_token_id(token_to_id, token);
}

std::optional<std::pair<std::uint32_t, std::size_t>> find_wordpiece_trie_match(
    std::string_view piece,
    std::size_t start,
    const std::vector<detail::WordPieceConfig::TrieNode> & trie) {
  if (trie.empty()) {
    return std::nullopt;
  }

  std::size_t node_index = 0;
  std::optional<std::pair<std::uint32_t, std::size_t>> best;
  for (std::size_t end = start; end < piece.size(); ++end) {
    const auto byte = static_cast<unsigned char>(piece[end]);
    const auto child = trie[node_index].children.find(byte);
    if (child == trie[node_index].children.end()) {
      break;
    }
    node_index = child->second;
    if (node_index >= trie.size()) {
      break;
    }
    if (trie[node_index].token_id) {
      best = std::make_pair(*trie[node_index].token_id, end + 1);
    }
  }
  return best;
}

std::vector<BpeToken> tokenize_wordpiece_piece(
    std::string_view piece,
    const std::vector<std::string> & id_to_token,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const detail::WordPieceConfig & config) {
  if (piece.empty()) {
    return {};
  }

  const auto unk_id =
      cached_or_required_token_id(token_to_id, config.unk_token, config.unk_id);
  if (count_codepoints(piece) > config.max_input_chars_per_word) {
    return {BpeToken{unk_id, config.unk_token, Offset{0, piece.size()}}};
  }

  std::vector<BpeToken> tokens;
  std::size_t start = 0;
  while (start < piece.size()) {
    const auto & trie =
        start == 0 ? config.initial_trie : config.continuation_trie;
    if (const auto matched = find_wordpiece_trie_match(piece, start, trie)) {
      const auto id = matched->first;
      const auto end = matched->second;
      const auto value = id < id_to_token.size()
          ? id_to_token[id]
          : std::string(piece.substr(start, end - start));
      tokens.push_back(BpeToken{id, value, Offset{start, end}});
      start = end;
      continue;
    }

    std::size_t end = piece.size();
    std::optional<BpeToken> current;

    while (start < end) {
      auto candidate = std::string(piece.substr(start, end - start));
      if (start > 0) {
        candidate = config.continuing_subword_prefix + candidate;
      }

      const auto found = token_to_id.find(candidate);
      if (found != token_to_id.end()) {
        current = BpeToken{found->second, candidate, Offset{start, end}};
        break;
      }
      end = previous_codepoint_start(piece, end);
    }

    if (!current) {
      return {BpeToken{unk_id, config.unk_token, Offset{0, piece.size()}}};
    }
    tokens.push_back(*current);
    start = current->offset.end;
  }

  return tokens;
}

BpeToken tokenize_wordlevel_piece(
    std::string_view piece,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const detail::WordLevelConfig & config) {
  const auto found = token_to_id.find(std::string(piece));
  if (found != token_to_id.end()) {
    return BpeToken{found->second, std::string(piece), Offset{0, piece.size()}};
  }

  const auto unk_id =
      cached_or_required_token_id(token_to_id, config.unk_token, config.unk_id);
  return BpeToken{unk_id, config.unk_token, Offset{0, piece.size()}};
}

std::vector<BpeToken> tokenize_unigram_piece_uncached(
    std::string_view piece,
    const std::vector<std::string> & id_to_token,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const detail::UnigramConfig & config) {
  if (piece.empty()) {
    return {};
  }
  if (config.scores.size() > id_to_token.size()) {
    throw std::runtime_error("Unigram scores exceed vocabulary size");
  }

  constexpr double kUnkPenalty = 10.0;
  const double unk_score = config.min_score - kUnkPenalty;
  const auto size = piece.size();
  const auto vocab_size = config.scores.size();
  std::vector<UnigramBestPathNode> best(size + 1);
  best[0].starts_at = 0;

  for (std::size_t starts_at = 0; starts_at < size;) {
    if (!best[starts_at].starts_at) {
      starts_at = next_codepoint(piece, starts_at);
      continue;
    }

    const auto mblen = next_codepoint(piece, starts_at) - starts_at;
    bool has_single_node = false;

    if (!config.trie.empty()) {
      std::size_t node_index = 0;
      for (std::size_t end = starts_at; end < size;) {
        const auto byte = static_cast<unsigned char>(piece[end]);
        const auto child = config.trie[node_index].children.find(byte);
        if (child == config.trie[node_index].children.end()) {
          break;
        }

        ++end;
        node_index = child->second;
        if (node_index >= config.trie.size()) {
          break;
        }

        for (const auto id : config.trie[node_index].token_ids) {
          if (id >= vocab_size) {
            continue;
          }
          auto & target = best[end];
          const auto candidate_score = best[starts_at].score + config.scores[id];
          if (!target.starts_at || candidate_score > target.score) {
            target.score = candidate_score;
            target.starts_at = starts_at;
            target.id = id;
          }
          if (!has_single_node && end - starts_at == mblen) {
            has_single_node = true;
          }
        }
      }
    } else {
      for (std::uint32_t id = 0; id < vocab_size; ++id) {
        const auto & token = id_to_token[id];
        if (token.empty() || starts_at + token.size() > size ||
            piece.substr(starts_at, token.size()) != token) {
          continue;
        }

        const auto key_pos = starts_at + token.size();
        auto & target = best[key_pos];
        const auto candidate_score = best[starts_at].score + config.scores[id];
        if (!target.starts_at || candidate_score > target.score) {
          target.score = candidate_score;
          target.starts_at = starts_at;
          target.id = id;
        }
        if (!has_single_node && token.size() == mblen) {
          has_single_node = true;
        }
      }
    }

    if (!has_single_node) {
      if (!config.unk_id) {
        throw std::runtime_error("Unigram encountered an unknown token without unk_id");
      }
      auto & target = best[starts_at + mblen];
      const auto candidate_score = best[starts_at].score + unk_score;
      if (!target.starts_at || candidate_score > target.score) {
        target.score = candidate_score;
        target.starts_at = starts_at;
        target.id = *config.unk_id;
      }
    }

    starts_at += mblen;
  }

  std::vector<UnigramSpan> spans;
  for (std::size_t ends_at = size; ends_at > 0;) {
    const auto & node = best[ends_at];
    if (!node.starts_at) {
      throw std::runtime_error("Unigram failed to find a best path");
    }
    spans.push_back(UnigramSpan{
        node.id,
        *node.starts_at,
        ends_at,
        config.unk_id && node.id == *config.unk_id});
    ends_at = *node.starts_at;
  }
  std::reverse(spans.begin(), spans.end());

  std::vector<UnigramSpan> fused_spans;
  fused_spans.reserve(spans.size());
  for (const auto & span : spans) {
    if (config.fuse_unk && span.is_unk && !fused_spans.empty() &&
        fused_spans.back().is_unk && fused_spans.back().end == span.start) {
      fused_spans.back().end = span.end;
      continue;
    }
    fused_spans.push_back(span);
  }

  std::vector<BpeToken> tokens;
  for (const auto & span : fused_spans) {
    const auto value = std::string(piece.substr(span.start, span.end - span.start));
    const auto found = token_to_id.find(value);
    if (found != token_to_id.end()) {
      tokens.push_back(BpeToken{found->second, value, Offset{span.start, span.end}});
      continue;
    }

    if (config.byte_fallback) {
      std::vector<BpeToken> fallback_tokens;
      bool found_all_fallback_bytes = true;
      for (const auto byte : value) {
        const auto byte_index = static_cast<unsigned char>(byte);
        std::optional<std::uint32_t> fallback_id =
            config.byte_fallback_ids[byte_index];
        if (!fallback_id) {
          const auto fallback_token = bpe_byte_fallback_token(byte_index);
          const auto fallback_found = token_to_id.find(fallback_token);
          if (fallback_found == token_to_id.end()) {
            found_all_fallback_bytes = false;
            break;
          }
          fallback_id = fallback_found->second;
        }
        const auto fallback_token = *fallback_id < id_to_token.size()
            ? id_to_token[*fallback_id]
            : bpe_byte_fallback_token(byte_index);
        fallback_tokens.push_back(
            BpeToken{
                *fallback_id,
                fallback_token,
                Offset{span.start, span.end}});
      }
      if (found_all_fallback_bytes) {
        tokens.insert(tokens.end(), fallback_tokens.begin(), fallback_tokens.end());
        continue;
      }
    }

    if (!config.unk_id) {
      throw std::runtime_error("Unigram encountered an unknown token without unk_id");
    }
    tokens.push_back(BpeToken{*config.unk_id, value, Offset{span.start, span.end}});
  }

  return tokens;
}

bool can_cache_unigram_piece(std::uint64_t cache_id, std::string_view piece) {
  return cache_id != 0 && piece.size() < kUnigramCacheMaxLength;
}

std::vector<BpeToken> tokenize_unigram_piece(
    std::string_view piece,
    const std::vector<std::string> & id_to_token,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const detail::UnigramConfig & config,
    std::uint64_t cache_id) {
  if (!can_cache_unigram_piece(cache_id, piece)) {
    return tokenize_unigram_piece_uncached(piece, id_to_token, token_to_id, config);
  }

  auto & cache = unigram_thread_cache[cache_id];
  const auto key = std::string(piece);
  const auto hit = cache.find(key);
  if (hit != cache.end()) {
    return hit->second;
  }

  auto tokens = tokenize_unigram_piece_uncached(piece, id_to_token, token_to_id, config);
  if (cache.size() < kUnigramCacheCapacity) {
    cache.emplace(key, tokens);
  }
  return tokens;
}

bool is_token_space(std::string_view token, std::size_t pos, std::size_t length) {
  const auto current = token.substr(pos, length);
  return current == byte_level_space() ||
      (length == 1 && std::isspace(static_cast<unsigned char>(token[pos])) != 0);
}

std::size_t count_leading_token_spaces(std::string_view token) {
  std::size_t count = 0;
  for (std::size_t pos = 0; pos < token.size();) {
    const auto length = utf8_codepoint_length(static_cast<unsigned char>(token[pos]));
    if (!is_token_space(token, pos, length)) {
      break;
    }
    ++count;
    pos += length;
  }
  return count;
}

std::size_t count_trailing_token_spaces(std::string_view token) {
  std::vector<std::pair<std::size_t, std::size_t>> codepoints;
  for (std::size_t pos = 0; pos < token.size();) {
    const auto length = utf8_codepoint_length(static_cast<unsigned char>(token[pos]));
    codepoints.push_back({pos, length});
    pos += length;
  }

  std::size_t count = 0;
  for (auto it = codepoints.rbegin(); it != codepoints.rend(); ++it) {
    if (!is_token_space(token, it->first, it->second)) {
      break;
    }
    ++count;
  }
  return count;
}

void process_byte_level_offsets(
    Encoding & encoding,
    const detail::ByteLevelConfig & config) {
  if (!config.enabled || !config.trim_offsets) {
    return;
  }

  for (std::size_t index = 0; index < encoding.tokens.size(); ++index) {
    auto & offset = encoding.offsets[index];
    auto leading_spaces = count_leading_token_spaces(encoding.tokens[index]);
    const auto trailing_spaces = count_trailing_token_spaces(encoding.tokens[index]);

    if (leading_spaces == 0 && trailing_spaces == 0) {
      continue;
    }

    if (leading_spaces > 0) {
      const bool is_first = index == 0 || offset.start == 0;
      if (is_first && config.add_prefix_space && leading_spaces == 1) {
        leading_spaces = 0;
      }
      offset.start = std::min(offset.start + leading_spaces, offset.end);
    }
    if (trailing_spaces > 0 && offset.end >= trailing_spaces) {
      offset.end = std::max(offset.end - trailing_spaces, offset.start);
    }
  }
}

void append_encoding_token(
    Encoding & encoding,
    std::uint32_t id,
    const std::string & token,
    Offset offset,
    std::uint32_t word_id,
    const std::vector<bool> & special_id) {
  encoding.ids.push_back(id);
  encoding.type_ids.push_back(0);
  encoding.tokens.push_back(token);
  encoding.offsets.push_back(offset);
  encoding.word_ids.push_back(word_id);
  (void)special_id;
  encoding.special_tokens_mask.push_back(0);
  encoding.attention_mask.push_back(1);
}

void append_special_encoding_token(
    Encoding & encoding,
    std::uint32_t id,
    const std::string & token,
    std::uint32_t type_id) {
  encoding.ids.push_back(id);
  encoding.type_ids.push_back(type_id);
  encoding.tokens.push_back(token);
  encoding.offsets.push_back(Offset{0, 0});
  encoding.word_ids.push_back(std::nullopt);
  encoding.special_tokens_mask.push_back(1);
  encoding.attention_mask.push_back(1);
}

detail::ProcessingPiece sequence_piece(
    detail::ProcessingPieceKind kind,
    std::uint32_t type_id) {
  detail::ProcessingPiece piece;
  piece.kind = kind;
  piece.type_id = type_id;
  return piece;
}

detail::ProcessingPiece special_piece(
    const std::string & token,
    std::uint32_t id,
    std::uint32_t type_id) {
  detail::ProcessingPiece piece;
  piece.kind = detail::ProcessingPieceKind::Special;
  piece.type_id = type_id;
  piece.ids.push_back(id);
  piece.tokens.push_back(token);
  return piece;
}

void append_sequence_piece(
    Encoding & output,
    const Encoding & input,
    std::uint32_t type_id) {
  for (std::size_t index = 0; index < input.ids.size(); ++index) {
    output.ids.push_back(input.ids[index]);
    output.type_ids.push_back(type_id);
    output.tokens.push_back(input.tokens[index]);
    output.offsets.push_back(input.offsets[index]);
    output.word_ids.push_back(input.word_ids[index]);
    output.special_tokens_mask.push_back(0);
    output.attention_mask.push_back(input.attention_mask[index]);
  }
}

void append_special_piece(Encoding & output, const detail::ProcessingPiece & piece) {
  if (piece.ids.size() != piece.tokens.size()) {
    throw std::runtime_error("post-processor special token ids/tokens size mismatch");
  }
  for (std::size_t index = 0; index < piece.ids.size(); ++index) {
    append_special_encoding_token(
        output,
        piece.ids[index],
        piece.tokens[index],
        piece.type_id);
  }
}

Encoding apply_processing_template(
    const Encoding & first,
    const Encoding * second,
    const std::vector<detail::ProcessingPiece> & pieces,
    bool add_special_tokens) {
  Encoding processed;
  for (const auto & piece : pieces) {
    if (piece.kind == detail::ProcessingPieceKind::SequenceA) {
      append_sequence_piece(processed, first, piece.type_id);
    } else if (piece.kind == detail::ProcessingPieceKind::SequenceB) {
      if (second != nullptr) {
        append_sequence_piece(processed, *second, piece.type_id);
      }
    } else if (add_special_tokens) {
      append_special_piece(processed, piece);
    }
  }
  return processed;
}

Encoding apply_bert_processing_single(
    Encoding encoding,
    const detail::BertProcessingConfig & config,
    bool add_special_tokens) {
  if (!config.enabled) {
    return encoding;
  }

  auto overflows = std::move(encoding.overflowing);
  encoding.overflowing.clear();
  Encoding processed = apply_processing_template(
      encoding,
      nullptr,
      config.single_template,
      add_special_tokens);
  for (auto & overflow : overflows) {
    processed.overflowing.push_back(apply_bert_processing_single(
        std::move(overflow),
        config,
        add_special_tokens));
  }
  return processed;
}

Encoding apply_bert_processing_pair(
    Encoding first,
    Encoding second,
    const detail::BertProcessingConfig & config,
    bool add_special_tokens) {
  const auto default_pair_template = std::vector<detail::ProcessingPiece>{
      sequence_piece(detail::ProcessingPieceKind::SequenceA, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceB, 1),
  };
  const auto & pieces = config.enabled ? config.pair_template : default_pair_template;

  auto merge_pair = [&](Encoding first_part, Encoding second_part) {
    auto first_overflows = std::move(first_part.overflowing);
    auto second_overflows = std::move(second_part.overflowing);
    first_part.overflowing.clear();
    second_part.overflowing.clear();

    Encoding merged = apply_processing_template(
        first_part,
        &second_part,
        pieces,
        add_special_tokens);

    for (const auto & first_overflow : first_overflows) {
      Encoding overflow_with_second = apply_processing_template(
          first_overflow,
          &second_part,
          pieces,
          add_special_tokens);
      std::vector<Encoding> both_overflows;

      for (const auto & second_overflow : second_overflows) {
        Encoding both_overflow = apply_processing_template(
            first_overflow,
            &second_overflow,
            pieces,
            add_special_tokens);
        overflow_with_second.overflowing.push_back(both_overflow);
        both_overflows.push_back(std::move(both_overflow));
      }

      merged.overflowing.push_back(std::move(overflow_with_second));

      for (auto & both_overflow : both_overflows) {
        merged.overflowing.push_back(std::move(both_overflow));
      }
    }

    for (const auto & second_overflow : second_overflows) {
      Encoding overflow_with_first = apply_processing_template(
          first_part,
          &second_overflow,
          pieces,
          add_special_tokens);
      for (const auto & first_overflow : first_overflows) {
        overflow_with_first.overflowing.push_back(apply_processing_template(
            first_overflow,
            &second_overflow,
            pieces,
            add_special_tokens));
      }
      merged.overflowing.push_back(std::move(overflow_with_first));
    }

    return merged;
  };

  return merge_pair(std::move(first), std::move(second));
}

Encoding slice_encoding(
    const Encoding & encoding,
    std::size_t start,
    std::size_t stop) {
  Encoding sliced;
  sliced.ids.assign(encoding.ids.begin() + start, encoding.ids.begin() + stop);
  sliced.type_ids.assign(
      encoding.type_ids.begin() + start,
      encoding.type_ids.begin() + stop);
  sliced.tokens.assign(
      encoding.tokens.begin() + start,
      encoding.tokens.begin() + stop);
  sliced.offsets.assign(
      encoding.offsets.begin() + start,
      encoding.offsets.begin() + stop);
  sliced.word_ids.assign(
      encoding.word_ids.begin() + start,
      encoding.word_ids.begin() + stop);
  sliced.special_tokens_mask.assign(
      encoding.special_tokens_mask.begin() + start,
      encoding.special_tokens_mask.begin() + stop);
  sliced.attention_mask.assign(
      encoding.attention_mask.begin() + start,
      encoding.attention_mask.begin() + stop);
  return sliced;
}

std::vector<std::pair<std::size_t, std::size_t>> truncation_ranges(
    std::size_t encoding_len,
    std::size_t max_len,
    std::size_t stride,
    detail::TruncationDirection direction) {
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  const auto step = max_len - stride;
  bool end = false;

  if (direction == detail::TruncationDirection::Right) {
    for (std::size_t start = 0; start < encoding_len && !end; start += step) {
      const auto stop = std::min(start + max_len, encoding_len);
      end = stop == encoding_len;
      ranges.push_back({start, stop});
    }
    return ranges;
  }

  for (std::size_t stop = encoding_len; stop > 0 && !end;) {
    const auto start = stop > max_len ? stop - max_len : 0;
    end = start == 0;
    ranges.push_back({start, stop});
    if (stop <= step) {
      break;
    }
    stop -= step;
  }
  return ranges;
}

void truncate_encoding(
    Encoding & encoding,
    const detail::TruncationConfig & config) {
  if (!config.enabled || config.max_length >= encoding.ids.size()) {
    return;
  }

  if (config.max_length == 0) {
    Encoding original = std::move(encoding);
    encoding = Encoding{};
    encoding.overflowing.push_back(std::move(original));
    return;
  }

  if (config.stride >= config.max_length) {
    throw std::runtime_error("truncation stride must be smaller than max_length");
  }

  const auto ranges = truncation_ranges(
      encoding.ids.size(),
      config.max_length,
      config.stride,
      config.direction);
  if (ranges.empty()) {
    return;
  }

  Encoding truncated = slice_encoding(encoding, ranges.front().first, ranges.front().second);
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    truncated.overflowing.push_back(
        slice_encoding(encoding, ranges[index].first, ranges[index].second));
  }
  encoding = std::move(truncated);
}

std::size_t special_tokens_added_for_single(
    const detail::BertProcessingConfig & special_processing,
    bool add_special_tokens) {
  if (!special_processing.enabled || !add_special_tokens) {
    return 0;
  }
  std::size_t added_tokens = 0;
  for (const auto & piece : special_processing.single_template) {
    if (piece.kind == detail::ProcessingPieceKind::Special) {
      added_tokens += piece.ids.size();
    }
  }
  return added_tokens;
}

void truncate_single_before_processing(
    Encoding & encoding,
    const detail::TruncationConfig & truncation,
    const detail::BertProcessingConfig & special_processing,
    bool add_special_tokens) {
  if (!truncation.enabled) {
    return;
  }

  auto effective = truncation;
  const auto added_tokens =
      special_tokens_added_for_single(special_processing, add_special_tokens);
  effective.max_length = added_tokens >= effective.max_length
      ? 0
      : effective.max_length - added_tokens;
  if (effective.max_length > 0 &&
      effective.strategy == detail::TruncationStrategy::OnlySecond &&
      encoding.ids.size() > effective.max_length) {
    throw std::runtime_error(
        "truncation strategy OnlySecond requires a pair sequence");
  }
  truncate_encoding(encoding, effective);
}

std::size_t special_tokens_added_for_pair(
    const detail::BertProcessingConfig & special_processing,
    bool add_special_tokens) {
  if (!special_processing.enabled || !add_special_tokens) {
    return 0;
  }
  std::size_t added_tokens = 0;
  for (const auto & piece : special_processing.pair_template) {
    if (piece.kind == detail::ProcessingPieceKind::Special) {
      added_tokens += piece.ids.size();
    }
  }
  return added_tokens;
}

void truncate_pair_before_processing(
    Encoding & first,
    Encoding & second,
    const detail::TruncationConfig & truncation,
    const detail::BertProcessingConfig & special_processing,
    bool add_special_tokens) {
  if (!truncation.enabled) {
    return;
  }

  auto effective = truncation;
  const auto added_tokens =
      special_tokens_added_for_pair(special_processing, add_special_tokens);
  effective.max_length = added_tokens >= effective.max_length
      ? 0
      : effective.max_length - added_tokens;

  if (effective.max_length == 0) {
    truncate_encoding(first, effective);
    truncate_encoding(second, effective);
    return;
  }
  if (effective.stride >= effective.max_length) {
    throw std::runtime_error("truncation stride must be smaller than max_length");
  }

  const auto total_length = first.ids.size() + second.ids.size();
  if (total_length <= effective.max_length) {
    return;
  }
  const auto to_remove = total_length - effective.max_length;

  if (effective.strategy == detail::TruncationStrategy::LongestFirst) {
    auto first_max = first.ids.size();
    auto second_max = second.ids.size();
    bool swapped = false;
    if (first_max > second_max) {
      swapped = true;
      std::swap(first_max, second_max);
    }

    if (first_max > effective.max_length) {
      second_max = first_max;
    } else {
      second_max = std::max(first_max, effective.max_length - first_max);
    }

    if (first_max + second_max > effective.max_length) {
      first_max = effective.max_length / 2;
      second_max = first_max + effective.max_length % 2;
    }

    if (swapped) {
      std::swap(first_max, second_max);
    }

    auto first_config = effective;
    first_config.max_length = first_max;
    truncate_encoding(first, first_config);

    auto second_config = effective;
    second_config.max_length = second_max;
    truncate_encoding(second, second_config);
    return;
  }

  auto & target = effective.strategy == detail::TruncationStrategy::OnlyFirst
      ? first
      : second;
  const auto target_length = target.ids.size();
  if (target_length <= to_remove) {
    throw std::runtime_error(
        "truncation target sequence is too short for max_length");
  }

  auto target_config = effective;
  target_config.max_length = target_length - to_remove;
  truncate_encoding(target, target_config);
}

std::size_t padding_target_length(
    const Encoding & encoding,
    const detail::PaddingConfig & config) {
  auto target_length = config.strategy == detail::PaddingStrategy::Fixed
      ? config.fixed_size
      : encoding.ids.size();

  if (config.pad_to_multiple_of && *config.pad_to_multiple_of > 0 &&
      target_length % *config.pad_to_multiple_of > 0) {
    target_length += *config.pad_to_multiple_of -
        (target_length % *config.pad_to_multiple_of);
  }

  return target_length;
}

void pad_encoding(
    Encoding & encoding,
    std::size_t target_length,
    const detail::PaddingConfig & config) {
  for (auto & overflow : encoding.overflowing) {
    pad_encoding(overflow, target_length, config);
  }

  if (encoding.ids.size() >= target_length) {
    return;
  }

  const auto pad_length = target_length - encoding.ids.size();
  if (config.direction == detail::PaddingDirection::Right) {
    encoding.ids.insert(encoding.ids.end(), pad_length, config.pad_id);
    encoding.type_ids.insert(encoding.type_ids.end(), pad_length, config.pad_type_id);
    encoding.tokens.insert(encoding.tokens.end(), pad_length, config.pad_token);
    encoding.offsets.insert(encoding.offsets.end(), pad_length, Offset{0, 0});
    encoding.word_ids.insert(encoding.word_ids.end(), pad_length, std::nullopt);
    encoding.special_tokens_mask.insert(
        encoding.special_tokens_mask.end(),
        pad_length,
        1);
    encoding.attention_mask.insert(encoding.attention_mask.end(), pad_length, 0);
    return;
  }

  encoding.ids.insert(encoding.ids.begin(), pad_length, config.pad_id);
  encoding.type_ids.insert(encoding.type_ids.begin(), pad_length, config.pad_type_id);
  encoding.tokens.insert(encoding.tokens.begin(), pad_length, config.pad_token);
  encoding.offsets.insert(encoding.offsets.begin(), pad_length, Offset{0, 0});
  encoding.word_ids.insert(encoding.word_ids.begin(), pad_length, std::nullopt);
  encoding.special_tokens_mask.insert(
      encoding.special_tokens_mask.begin(),
      pad_length,
      1);
  encoding.attention_mask.insert(encoding.attention_mask.begin(), pad_length, 0);
}

void apply_padding(Encoding & encoding, const detail::PaddingConfig & config) {
  if (!config.enabled) {
    return;
  }
  pad_encoding(encoding, padding_target_length(encoding, config), config);
}

std::size_t batch_padding_target_length(
    const std::vector<Encoding> & encodings,
    const detail::PaddingConfig & config) {
  std::size_t target_length = 0;
  if (config.strategy == detail::PaddingStrategy::Fixed) {
    target_length = config.fixed_size;
  } else {
    for (const auto & encoding : encodings) {
      target_length = std::max(target_length, encoding.ids.size());
    }
  }

  if (config.pad_to_multiple_of && *config.pad_to_multiple_of > 0 &&
      target_length % *config.pad_to_multiple_of > 0) {
    target_length += *config.pad_to_multiple_of -
        (target_length % *config.pad_to_multiple_of);
  }
  return target_length;
}

void apply_padding_to_batch(
    std::vector<Encoding> & encodings,
    const detail::PaddingConfig & config) {
  if (!config.enabled || encodings.empty()) {
    return;
  }
  const auto target_length = batch_padding_target_length(encodings, config);
  for (auto & encoding : encodings) {
    pad_encoding(encoding, target_length, config);
  }
}

detail::BertProcessingConfig effective_special_processing(
    const detail::BertProcessingConfig & bert,
    const detail::TemplateProcessingConfig & template_processing) {
  if (bert.enabled || !template_processing.enabled) {
    return bert;
  }

  detail::BertProcessingConfig config;
  config.enabled = true;
  config.sep_token = template_processing.sep_token;
  config.sep_id = template_processing.sep_id;
  config.cls_token = template_processing.cls_token;
  config.cls_id = template_processing.cls_id;
  config.single_template = template_processing.single_template;
  config.pair_template = template_processing.pair_template;
  return config;
}

std::uint32_t resolve_word_id(
    std::optional<std::uint32_t> fixed_word_id,
    std::uint32_t & next_word_id) {
  if (fixed_word_id) {
    return *fixed_word_id;
  }
  return next_word_id++;
}

std::uint32_t current_word_id(
    std::optional<std::uint32_t> fixed_word_id,
    std::uint32_t next_word_id) {
  return fixed_word_id.value_or(next_word_id);
}

void finish_current_word(
    std::optional<std::uint32_t> fixed_word_id,
    std::uint32_t & next_word_id) {
  if (!fixed_word_id) {
    ++next_word_id;
  }
}

void encode_input_split(
    Encoding & encoding,
    const InputSplit & split,
    const std::vector<std::string> & id_to_token,
    const std::vector<bool> & special_id,
    const std::string & model_type,
    const HashMap<std::string, std::uint32_t> & token_to_id_map,
    const HashMap<std::uint64_t, detail::BpeMerge> & bpe_merges,
    const detail::BpeConfig & bpe,
    std::uint64_t bpe_cache_id,
    const detail::WordPieceConfig & wordpiece,
    const detail::WordLevelConfig & wordlevel,
    const detail::UnigramConfig & unigram,
    std::uint64_t unigram_cache_id,
    bool byte_level_normalizer,
    const detail::SimpleNormalizerConfig & simple_normalizer,
    const detail::BertNormalizerConfig & bert_normalizer,
    bool bert_pre_tokenizer,
    bool whitespace_pre_tokenizer,
    bool whitespace_split_pre_tokenizer,
    const detail::MetaspaceConfig & metaspace_pre_tokenizer,
    const detail::SplitPreTokenizerConfig & split_pre_tokenizer,
    const std::vector<detail::PreTokenizerStepConfig> & pre_tokenizer_steps,
    const detail::ByteLevelConfig & byte_level_pre_tokenizer,
    std::optional<std::uint32_t> fixed_word_id,
    std::uint32_t & next_word_id) {
  if (split.is_added_token) {
    const auto & token_text = split.token_text.empty() ? split.text : split.token_text;
    append_encoding_token(
        encoding,
        split.id,
        token_text,
        split.offset,
        resolve_word_id(fixed_word_id, next_word_id),
        special_id);
    return;
  }

  if (model_type == "BPE" && byte_level_normalizer &&
      !byte_level_pre_tokenizer.enabled) {
    const auto normalized = byte_level_normalize_split(split);
    for (const auto & piece : split_byte_level_normalized_piece(
             normalized,
             split_pre_tokenizer)) {
      const auto word_id = current_word_id(fixed_word_id, next_word_id);
      const auto tokens = tokenize_bpe_piece(
          piece,
          id_to_token,
          token_to_id_map,
          bpe_merges,
          bpe,
          bpe_cache_id);
      for (const auto & token : tokens) {
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            token.offset,
            word_id,
            special_id);
      }
      finish_current_word(fixed_word_id, next_word_id);
    }
    return;
  }

  if (model_type == "BPE" && byte_level_pre_tokenizer.enabled) {
    const auto normalized = simple_normalize_input(split, simple_normalizer);
    const auto normalized_piece = NormalizedPiece{
        normalized.text,
        normalized.normalized_byte_spans};
    for (const auto & pre_split : split_normalized_piece(
             normalized_piece,
             split_pre_tokenizer)) {
      for (const auto & piece : byte_level_pre_tokenize(
               pre_split,
               byte_level_pre_tokenizer)) {
        const auto word_id = current_word_id(fixed_word_id, next_word_id);
        const auto tokens = tokenize_bpe_piece(
            piece,
            id_to_token,
            token_to_id_map,
            bpe_merges,
            bpe,
            bpe_cache_id);
        for (const auto & token : tokens) {
          append_encoding_token(
              encoding,
              token.id,
              token.value,
              token.offset,
              word_id,
              special_id);
        }
        finish_current_word(fixed_word_id, next_word_id);
      }
    }
    return;
  }

  if (model_type == "BPE") {
    const auto normalized = simple_normalize_input(split, simple_normalizer);
    const auto normalized_piece = NormalizedPiece{
        normalized.text,
        normalized.normalized_byte_spans};
    const auto pre_splits = pre_tokenizer_steps.empty()
        ? split_normalized_piece(normalized_piece, split_pre_tokenizer)
        : apply_pre_tokenizer_steps(normalized_piece, pre_tokenizer_steps);
    for (const auto & pre_split : pre_splits) {
      const auto word_id = current_word_id(fixed_word_id, next_word_id);
      const auto tokens = tokenize_bpe_piece(
          ByteLevelPiece{pre_split.text, pre_split.normalized_byte_spans},
          id_to_token,
          token_to_id_map,
          bpe_merges,
          bpe,
          bpe_cache_id);
      for (const auto & token : tokens) {
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            token.offset,
            word_id,
            special_id);
      }
      finish_current_word(fixed_word_id, next_word_id);
    }
    return;
  }

  if (model_type == "WordPiece") {
    const auto normalized = simple_normalizer.enabled
        ? simple_normalize_input(split, simple_normalizer)
        : bert_normalize_input(split, bert_normalizer);
    const auto pieces = bert_pre_tokenizer
        ? bert_pre_tokenize(normalized.text)
        : (whitespace_pre_tokenizer
              ? whitespace_regex_pre_tokenize(normalized.text)
              : whitespace_pre_tokenize(normalized.text));

    for (const auto & piece : pieces) {
      const auto word_id = current_word_id(fixed_word_id, next_word_id);
      const auto piece_text =
          std::string_view(normalized.text).substr(piece.start, piece.end - piece.start);
      const auto tokens = tokenize_wordpiece_piece(
          piece_text,
          id_to_token,
          token_to_id_map,
          wordpiece);
      for (const auto & token : tokens) {
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            convert_normalized_range(
                normalized,
                piece.start + token.offset.start,
                piece.start + token.offset.end),
            word_id,
            special_id);
      }
      finish_current_word(fixed_word_id, next_word_id);
    }
    return;
  }

  if (model_type == "Unigram") {
    const auto normalized = simple_normalize_input(split, simple_normalizer);
    const auto pieces = sentencepiece_pre_tokenize(
        normalized,
        whitespace_split_pre_tokenizer,
        metaspace_pre_tokenizer);
    for (const auto & piece : pieces) {
      const auto word_id = current_word_id(fixed_word_id, next_word_id);
      const auto tokens = tokenize_unigram_piece(
          piece.text,
          id_to_token,
          token_to_id_map,
          unigram,
          unigram_cache_id);
      for (const auto & token : tokens) {
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            convert_piece_offsets(piece, token.offset.start, token.offset.end),
            word_id,
            special_id);
      }
      finish_current_word(fixed_word_id, next_word_id);
    }
    return;
  }

  if (model_type == "WordLevel") {
    if (!pre_tokenizer_steps.empty()) {
      for (const auto & pre_split : apply_pre_tokenizer_steps(
               split,
               pre_tokenizer_steps)) {
        const auto token = tokenize_wordlevel_piece(
            std::string_view(pre_split.text),
            token_to_id_map,
            wordlevel);
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            Offset{
                pre_split.offset.start + token.offset.start,
                pre_split.offset.start + token.offset.end},
            resolve_word_id(fixed_word_id, next_word_id),
            special_id);
      }
      return;
    }

    if (split_pre_tokenizer.enabled) {
      for (const auto & pre_split : split_pre_tokenize(split, split_pre_tokenizer)) {
        const auto token = tokenize_wordlevel_piece(
            std::string_view(pre_split.text),
            token_to_id_map,
            wordlevel);
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            Offset{
                pre_split.offset.start + token.offset.start,
                pre_split.offset.start + token.offset.end},
            resolve_word_id(fixed_word_id, next_word_id),
            special_id);
      }
      return;
    }

    if (whitespace_pre_tokenizer) {
      for (const auto & piece : whitespace_regex_pre_tokenize(split.text)) {
        const auto token = tokenize_wordlevel_piece(
            std::string_view(split.text).substr(piece.start, piece.end - piece.start),
            token_to_id_map,
            wordlevel);
        append_encoding_token(
            encoding,
            token.id,
            token.value,
            Offset{
                split.offset.start + piece.start + token.offset.start,
                split.offset.start + piece.start + token.offset.end},
            resolve_word_id(fixed_word_id, next_word_id),
            special_id);
      }
      return;
    }

    for (const auto & [piece, relative_offset] : split_on_ascii_whitespace(split.text)) {
      const auto token = tokenize_wordlevel_piece(piece, token_to_id_map, wordlevel);
      append_encoding_token(
          encoding,
          token.id,
          token.value,
          Offset{
              split.offset.start + relative_offset.start + token.offset.start,
              split.offset.start + relative_offset.start + token.offset.end},
          resolve_word_id(fixed_word_id, next_word_id),
          special_id);
    }
    return;
  }

  for (const auto & [piece, relative_offset] : split_on_ascii_whitespace(split.text)) {
    const auto found = token_to_id_map.find(piece);
    if (found == token_to_id_map.end()) {
      continue;
    }
    append_encoding_token(
        encoding,
        found->second,
        piece,
        Offset{
            split.offset.start + relative_offset.start,
            split.offset.start + relative_offset.end},
        resolve_word_id(fixed_word_id, next_word_id),
        special_id);
  }
}

std::optional<AddedTokenMatch> find_next_added_token_match(
    const std::vector<detail::AddedToken> & added_tokens,
    std::string_view text,
    std::size_t search_start) {
  std::optional<AddedTokenMatch> best;

  for (std::size_t token_index = 0; token_index < added_tokens.size(); ++token_index) {
    const auto & token = added_tokens[token_index];
    if (token.content.empty()) {
      continue;
    }

    const auto found = text.find(token.content, search_start);
    if (found == std::string_view::npos) {
      continue;
    }
    const auto stop = found + token.content.size();
    const auto found_length = stop - found;
    const auto best_length = best ? best->stop - best->start : 0;
    if (!best || found < best->start ||
        (found == best->start && found_length > best_length)) {
      best = AddedTokenMatch{token_index, found, stop};
    }
  }

  return best;
}

std::optional<AddedTokenMatch> find_next_added_token_match(
    const std::vector<AddedTokenMatch> & matches,
    std::size_t search_start,
    std::size_t & cursor) {
  while (cursor < matches.size() && matches[cursor].start < search_start) {
    ++cursor;
  }
  if (cursor == matches.size()) {
    return std::nullopt;
  }
  return matches[cursor];
}

std::shared_ptr<AddedTokenMatcher> build_added_token_matcher(
    const std::vector<detail::AddedToken> & added_tokens) {
  if (added_tokens.empty()) {
    return nullptr;
  }
  auto matcher = std::make_shared<AddedTokenMatcher>(added_tokens);
  if (!matcher->use_trie()) {
    return nullptr;
  }
  return matcher;
}

void rebuild_added_token_matcher(
    const std::vector<detail::AddedToken> & added_tokens,
    std::shared_ptr<AddedTokenMatcher> & matcher) {
  matcher = build_added_token_matcher(added_tokens);
}

std::vector<InputSplit> split_on_added_tokens(
    const std::string & text,
    const std::vector<detail::AddedToken> & added_tokens,
    AddedTokenMatcher * matcher) {
  if (added_tokens.empty()) {
    return {InputSplit{false, 0, text, Offset{0, text.size()}}};
  }

  const auto matcher_matches = matcher ? matcher->find_matches(text) : std::vector<AddedTokenMatch>{};

  std::vector<InputSplit> splits;
  std::size_t start_offset = 0;
  std::size_t search_start = 0;
  std::size_t matcher_cursor = 0;

  while (search_start < text.size()) {
    const auto match = matcher
        ? find_next_added_token_match(matcher_matches, search_start, matcher_cursor)
        : find_next_added_token_match(added_tokens, text, search_start);
    if (!match) {
      break;
    }

    const auto & added_token = added_tokens[match->token_index];
    auto start = match->start;
    auto stop = match->stop;
    search_start = stop;

    if (added_token.single_word) {
      const bool left_boundary = start == 0 || !has_word_byte_before(text, start);
      const bool right_boundary = stop == text.size() || !has_word_byte_at(text, stop);
      if (!left_boundary || !right_boundary) {
        continue;
      }
    }

    if (added_token.lstrip) {
      start = std::max(whitespace_start_before(text, start), start_offset);
    }
    if (added_token.rstrip) {
      stop = whitespace_end_after(text, stop);
    }
    if (stop <= start_offset) {
      continue;
    }

    if (start_offset < start) {
      splits.push_back(InputSplit{
          false,
          0,
          text.substr(start_offset, start - start_offset),
          Offset{start_offset, start}});
    }

    splits.push_back(InputSplit{
        true,
        added_token.id,
        text.substr(start, stop - start),
        Offset{start, stop},
        added_token.normalized_content.empty()
            ? std::string{}
            : added_token.normalized_content});
    start_offset = stop;
    search_start = stop;
  }

  if (start_offset < text.size()) {
    splits.push_back(InputSplit{
        false,
        0,
        text.substr(start_offset),
        Offset{start_offset, text.size()}});
  }

  return splits;
}

const std::unordered_set<std::string> & supported_model_types() {
  static const std::unordered_set<std::string> types = {
      "BPE",
      "WordPiece",
      "WordLevel",
      "Unigram",
  };
  return types;
}

const std::unordered_set<std::string> & supported_normalizer_types() {
  static const std::unordered_set<std::string> types = {
      "BertNormalizer",
      "Strip",
      "StripAccents",
      "NFC",
      "NFD",
      "NFKC",
      "NFKD",
      "Sequence",
      "Lowercase",
      "Nmt",
      "Precompiled",
      "Replace",
      "Prepend",
      "ByteLevel",
  };
  return types;
}

const std::unordered_set<std::string> & supported_pre_tokenizer_types() {
  static const std::unordered_set<std::string> types = {
      "BertPreTokenizer",
      "ByteLevel",
      "CharDelimiterSplit",
      "Metaspace",
      "Whitespace",
      "Sequence",
      "Split",
      "Punctuation",
      "WhitespaceSplit",
      "Digits",
      "UnicodeScripts",
      "FixedLength",
  };
  return types;
}

const std::unordered_set<std::string> & supported_post_processor_types() {
  static const std::unordered_set<std::string> types = {
      "RobertaProcessing",
      "BertProcessing",
      "ByteLevel",
      "TemplateProcessing",
      "Sequence",
  };
  return types;
}

const std::unordered_set<std::string> & supported_decoder_types() {
  static const std::unordered_set<std::string> types = {
      "BPEDecoder",
      "ByteLevel",
      "WordPiece",
      "Metaspace",
      "CTC",
      "Sequence",
      "Replace",
      "Fuse",
      "Strip",
      "ByteFallback",
  };
  return types;
}

std::string require_wrapper_type(
    const json & wrapper,
    std::string_view slot,
    const std::unordered_set<std::string> & supported) {
  if (!wrapper.is_object()) {
    throw std::runtime_error(std::string(slot) + " wrapper must be an object");
  }
  if (!wrapper.contains("type") || !wrapper.at("type").is_string()) {
    throw std::runtime_error(std::string(slot) + " wrapper must contain a string type");
  }

  const auto type = wrapper.at("type").get<std::string>();
  if (supported.find(type) == supported.end()) {
    throw std::runtime_error(
        "unsupported " + std::string(slot) + " wrapper type: " + type);
  }
  return type;
}

bool is_uint32_json(const json & value) {
  return value.is_number_unsigned() &&
      value.get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max();
}

std::uint32_t require_uint32(const json & object, std::string_view field) {
  if (!object.contains(field) || !is_uint32_json(object.at(field))) {
    throw std::runtime_error(
        "model field " + std::string(field) + " must be an unsigned 32-bit integer");
  }
  return object.at(field).get<std::uint32_t>();
}

std::uint32_t require_added_token_id(const json & object) {
  if (!object.contains("id") || !is_uint32_json(object.at("id"))) {
    throw std::runtime_error(
        "added_tokens entries must contain an unsigned 32-bit id");
  }
  return object.at("id").get<std::uint32_t>();
}

bool optional_added_token_bool(
    const json & object,
    std::string_view field,
    bool default_value) {
  if (!object.contains(field)) {
    return default_value;
  }
  if (!object.at(field).is_boolean()) {
    throw std::runtime_error(
        "added_tokens field " + std::string(field) + " must be a boolean");
  }
  return object.at(field).get<bool>();
}

bool optional_wrapper_bool(
    const json & object,
    std::string_view wrapper_name,
    std::string_view field,
    bool default_value) {
  if (!object.contains(field)) {
    return default_value;
  }
  if (!object.at(field).is_boolean()) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be a boolean");
  }
  return object.at(field).get<bool>();
}

bool optional_model_bool(
    const json & object,
    std::string_view model_name,
    std::string_view field,
    bool default_value) {
  if (!object.contains(field)) {
    return default_value;
  }
  if (!object.at(field).is_boolean()) {
    throw std::runtime_error(
        std::string(model_name) + " model " + std::string(field) +
        " must be a boolean");
  }
  return object.at(field).get<bool>();
}

std::optional<std::string> optional_model_string_or_null(
    const json & object,
    std::string_view model_name,
    std::string_view field) {
  if (!object.contains(field) || object.at(field).is_null()) {
    return std::nullopt;
  }
  if (!object.at(field).is_string()) {
    throw std::runtime_error(
        std::string(model_name) + " model " + std::string(field) +
        " must be a string or null");
  }
  return object.at(field).get<std::string>();
}

std::optional<double> optional_model_probability_or_null(
    const json & object,
    std::string_view model_name,
    std::string_view field) {
  if (!object.contains(field) || object.at(field).is_null()) {
    return std::nullopt;
  }
  if (!object.at(field).is_number()) {
    throw std::runtime_error(
        std::string(model_name) + " model " + std::string(field) +
        " must be a number between 0 and 1 or null");
  }
  const auto value = object.at(field).get<double>();
  if (value < 0.0 || value > 1.0) {
    throw std::runtime_error(
        std::string(model_name) + " model " + std::string(field) +
        " must be between 0 and 1");
  }
  return value;
}

detail::AddedToken parse_added_token(const json & token) {
  if (!token.is_object()) {
    throw std::runtime_error("added_tokens entries must be objects");
  }
  if (!token.contains("content") || !token.at("content").is_string()) {
    throw std::runtime_error("added_tokens entries must contain string content");
  }

  detail::AddedToken added_token;
  added_token.id = require_added_token_id(token);
  added_token.content = token.at("content").get<std::string>();
  added_token.single_word = optional_added_token_bool(token, "single_word", false);
  added_token.lstrip = optional_added_token_bool(token, "lstrip", false);
  added_token.rstrip = optional_added_token_bool(token, "rstrip", false);
  added_token.special = optional_added_token_bool(token, "special", false);
  added_token.normalized =
      optional_added_token_bool(token, "normalized", !added_token.special);
  return added_token;
}

detail::ByteLevelConfig parse_byte_level_config(const json & wrapper) {
  detail::ByteLevelConfig config;
  config.enabled = true;
  config.add_prefix_space =
      optional_wrapper_bool(wrapper, "ByteLevel", "add_prefix_space", true);
  config.trim_offsets =
      optional_wrapper_bool(wrapper, "ByteLevel", "trim_offsets", true);
  config.use_regex =
      optional_wrapper_bool(wrapper, "ByteLevel", "use_regex", true);
  return config;
}

std::optional<bool> optional_nullable_wrapper_bool(
    const json & object,
    std::string_view wrapper_name,
    std::string_view field) {
  if (!object.contains(field) || object.at(field).is_null()) {
    return std::nullopt;
  }
  if (!object.at(field).is_boolean()) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be a boolean or null");
  }
  return object.at(field).get<bool>();
}

detail::BertNormalizerConfig parse_bert_normalizer_config(const json & wrapper) {
  detail::BertNormalizerConfig config;
  config.enabled = true;
  config.clean_text =
      optional_wrapper_bool(wrapper, "BertNormalizer", "clean_text", true);
  config.handle_chinese_chars =
      optional_wrapper_bool(wrapper, "BertNormalizer", "handle_chinese_chars", true);
  config.strip_accents =
      optional_nullable_wrapper_bool(wrapper, "BertNormalizer", "strip_accents");
  config.lowercase =
      optional_wrapper_bool(wrapper, "BertNormalizer", "lowercase", true);
  return config;
}

std::pair<std::string, std::uint32_t> parse_token_id_pair(
    const json & wrapper,
    std::string_view wrapper_name,
    std::string_view field) {
  if (!wrapper.contains(field) || !wrapper.at(field).is_array() ||
      wrapper.at(field).size() != 2 || !wrapper.at(field).at(0).is_string() ||
      !is_uint32_json(wrapper.at(field).at(1))) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be [string, uint32]");
  }
  return {
      wrapper.at(field).at(0).get<std::string>(),
      wrapper.at(field).at(1).get<std::uint32_t>()};
}

detail::BertProcessingConfig parse_bert_processing_config(const json & wrapper) {
  detail::BertProcessingConfig config;
  config.enabled = true;
  const auto sep = parse_token_id_pair(wrapper, "BertProcessing", "sep");
  const auto cls = parse_token_id_pair(wrapper, "BertProcessing", "cls");
  config.sep_token = sep.first;
  config.sep_id = sep.second;
  config.cls_token = cls.first;
  config.cls_id = cls.second;
  config.single_template = {
      special_piece(config.cls_token, config.cls_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceA, 0),
      special_piece(config.sep_token, config.sep_id, 0),
  };
  config.pair_template = {
      special_piece(config.cls_token, config.cls_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceA, 0),
      special_piece(config.sep_token, config.sep_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceB, 1),
      special_piece(config.sep_token, config.sep_id, 1),
  };
  return config;
}

detail::BertProcessingConfig parse_roberta_processing_config(const json & wrapper) {
  detail::BertProcessingConfig config;
  config.enabled = true;
  const auto sep = parse_token_id_pair(wrapper, "RobertaProcessing", "sep");
  const auto cls = parse_token_id_pair(wrapper, "RobertaProcessing", "cls");
  config.sep_token = sep.first;
  config.sep_id = sep.second;
  config.cls_token = cls.first;
  config.cls_id = cls.second;
  config.single_template = {
      special_piece(config.cls_token, config.cls_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceA, 0),
      special_piece(config.sep_token, config.sep_id, 0),
  };
  config.pair_template = {
      special_piece(config.cls_token, config.cls_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceA, 0),
      special_piece(config.sep_token, config.sep_id, 0),
      special_piece(config.sep_token, config.sep_id, 0),
      sequence_piece(detail::ProcessingPieceKind::SequenceB, 0),
      special_piece(config.sep_token, config.sep_id, 0),
  };
  config.offset_processor.enabled = true;
  config.offset_processor.add_prefix_space =
      optional_wrapper_bool(wrapper, "RobertaProcessing", "add_prefix_space", true);
  config.offset_processor.trim_offsets =
      optional_wrapper_bool(wrapper, "RobertaProcessing", "trim_offsets", true);
  config.offset_processor.use_regex = true;
  return config;
}

std::optional<std::string> optional_wrapper_string(
    const json & object,
    std::string_view wrapper_name,
    std::string_view field) {
  if (!object.contains(field) || object.at(field).is_null()) {
    return std::nullopt;
  }
  if (!object.at(field).is_string()) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be a string");
  }
  return object.at(field).get<std::string>();
}

detail::WordPieceDecoderConfig parse_wordpiece_decoder_config(const json & wrapper) {
  detail::WordPieceDecoderConfig config;
  config.enabled = true;
  config.prefix =
      optional_wrapper_string(wrapper, "WordPiece", "prefix").value_or("##");
  config.cleanup =
      optional_wrapper_bool(wrapper, "WordPiece", "cleanup", true);
  return config;
}

std::string ascii_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
  return value;
}

detail::TruncationDirection parse_truncation_direction(const json & wrapper) {
  if (!wrapper.contains("direction")) {
    return detail::TruncationDirection::Right;
  }
  if (!wrapper.at("direction").is_string()) {
    throw std::runtime_error("truncation direction must be a string");
  }

  const auto value = ascii_lower(wrapper.at("direction").get<std::string>());
  if (value == "right") {
    return detail::TruncationDirection::Right;
  }
  if (value == "left") {
    return detail::TruncationDirection::Left;
  }
  throw std::runtime_error("unsupported truncation direction: " + value);
}

detail::TruncationStrategy parse_truncation_strategy(const json & wrapper) {
  if (!wrapper.contains("strategy")) {
    return detail::TruncationStrategy::LongestFirst;
  }
  if (!wrapper.at("strategy").is_string()) {
    throw std::runtime_error("truncation strategy must be a string");
  }

  const auto value = ascii_lower(wrapper.at("strategy").get<std::string>());
  if (value == "longestfirst" || value == "longest_first") {
    return detail::TruncationStrategy::LongestFirst;
  }
  if (value == "onlyfirst" || value == "only_first") {
    return detail::TruncationStrategy::OnlyFirst;
  }
  if (value == "onlysecond" || value == "only_second") {
    return detail::TruncationStrategy::OnlySecond;
  }
  throw std::runtime_error("unsupported truncation strategy: " + value);
}

detail::PaddingDirection parse_padding_direction(const json & wrapper) {
  if (!wrapper.contains("direction") || !wrapper.at("direction").is_string()) {
    throw std::runtime_error("padding direction must be a string");
  }

  const auto value = ascii_lower(wrapper.at("direction").get<std::string>());
  if (value == "right") {
    return detail::PaddingDirection::Right;
  }
  if (value == "left") {
    return detail::PaddingDirection::Left;
  }
  throw std::runtime_error("unsupported padding direction: " + value);
}

void parse_padding_strategy(const json & wrapper, detail::PaddingConfig & config) {
  if (!wrapper.contains("strategy")) {
    throw std::runtime_error("padding strategy is required");
  }
  const auto & strategy = wrapper.at("strategy");
  if (strategy.is_string()) {
    const auto value = ascii_lower(strategy.get<std::string>());
    if (value == "batchlongest" || value == "batch_longest") {
      config.strategy = detail::PaddingStrategy::BatchLongest;
      return;
    }
    throw std::runtime_error("unsupported padding strategy: " + value);
  }
  if (strategy.is_object() && strategy.contains("Fixed") &&
      is_uint32_json(strategy.at("Fixed"))) {
    config.strategy = detail::PaddingStrategy::Fixed;
    config.fixed_size = strategy.at("Fixed").get<std::uint32_t>();
    return;
  }
  throw std::runtime_error("padding strategy must be BatchLongest or {Fixed:uint32}");
}

std::optional<detail::PaddingConfig> parse_padding_config(const json & root) {
  if (!root.contains("padding") || root.at("padding").is_null()) {
    return std::nullopt;
  }

  const auto & wrapper = root.at("padding");
  if (!wrapper.is_object()) {
    throw std::runtime_error("tokenizer JSON padding must be an object or null");
  }
  if (!wrapper.contains("pad_id") || !is_uint32_json(wrapper.at("pad_id"))) {
    throw std::runtime_error("padding pad_id must be an unsigned 32-bit integer");
  }
  if (!wrapper.contains("pad_type_id") ||
      !is_uint32_json(wrapper.at("pad_type_id"))) {
    throw std::runtime_error("padding pad_type_id must be an unsigned 32-bit integer");
  }
  if (!wrapper.contains("pad_token") || !wrapper.at("pad_token").is_string()) {
    throw std::runtime_error("padding pad_token must be a string");
  }
  if (wrapper.contains("pad_to_multiple_of") &&
      !wrapper.at("pad_to_multiple_of").is_null() &&
      !is_uint32_json(wrapper.at("pad_to_multiple_of"))) {
    throw std::runtime_error(
        "padding pad_to_multiple_of must be an unsigned 32-bit integer or null");
  }

  detail::PaddingConfig config;
  config.enabled = true;
  parse_padding_strategy(wrapper, config);
  config.direction = parse_padding_direction(wrapper);
  if (wrapper.contains("pad_to_multiple_of") &&
      !wrapper.at("pad_to_multiple_of").is_null()) {
    config.pad_to_multiple_of =
        wrapper.at("pad_to_multiple_of").get<std::uint32_t>();
  }
  config.pad_id = wrapper.at("pad_id").get<std::uint32_t>();
  config.pad_type_id = wrapper.at("pad_type_id").get<std::uint32_t>();
  config.pad_token = wrapper.at("pad_token").get<std::string>();
  return config;
}

std::optional<detail::TruncationConfig> parse_truncation_config(const json & root) {
  if (!root.contains("truncation") || root.at("truncation").is_null()) {
    return std::nullopt;
  }

  const auto & wrapper = root.at("truncation");
  if (!wrapper.is_object()) {
    throw std::runtime_error("tokenizer JSON truncation must be an object or null");
  }
  if (!wrapper.contains("max_length") || !is_uint32_json(wrapper.at("max_length"))) {
    throw std::runtime_error("truncation max_length must be an unsigned 32-bit integer");
  }
  if (!wrapper.contains("strategy") || !wrapper.at("strategy").is_string()) {
    throw std::runtime_error("truncation strategy must be a string");
  }
  if (!wrapper.contains("stride") || !is_uint32_json(wrapper.at("stride"))) {
    throw std::runtime_error("truncation stride must be an unsigned 32-bit integer");
  }

  detail::TruncationConfig config;
  config.enabled = true;
  config.max_length = wrapper.at("max_length").get<std::uint32_t>();
  config.stride = wrapper.at("stride").get<std::uint32_t>();
  config.direction = parse_truncation_direction(wrapper);
  config.strategy = parse_truncation_strategy(wrapper);
  if (config.max_length > 0 && config.stride >= config.max_length) {
    throw std::runtime_error("truncation stride must be smaller than max_length");
  }
  return config;
}

detail::MetaspaceConfig parse_metaspace_config(const json & wrapper) {
  detail::MetaspaceConfig config;
  config.enabled = true;
  config.replacement =
      optional_wrapper_string(wrapper, "Metaspace", "replacement").value_or("\xE2\x96\x81");
  config.split = optional_wrapper_bool(wrapper, "Metaspace", "split", true);
  config.prepend_scheme =
      optional_wrapper_string(wrapper, "Metaspace", "prepend_scheme").value_or("always");
  if (wrapper.contains("add_prefix_space")) {
    const auto add_prefix_space =
        optional_wrapper_bool(wrapper, "Metaspace", "add_prefix_space", true);
    if (!add_prefix_space) {
      if (config.prepend_scheme != "never") {
        throw std::runtime_error(
            "Metaspace add_prefix_space does not match declared prepend_scheme");
      }
      config.prepend_scheme = "never";
    }
  }
  if (config.prepend_scheme != "always" && config.prepend_scheme != "first" &&
      config.prepend_scheme != "never") {
    throw std::runtime_error("Metaspace prepend_scheme must be always, first, or never");
  }
  return config;
}

std::uint32_t require_wrapper_uint32(
    const json & wrapper,
    std::string_view wrapper_name,
    std::string_view field) {
  if (!wrapper.contains(field) || !is_uint32_json(wrapper.at(field))) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be an unsigned 32-bit integer");
  }
  return wrapper.at(field).get<std::uint32_t>();
}

std::string require_single_codepoint_string(
    const json & wrapper,
    std::string_view wrapper_name,
    std::string_view field) {
  if (!wrapper.contains(field) || !wrapper.at(field).is_string()) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be a one-character string");
  }
  const auto value = wrapper.at(field).get<std::string>();
  if (value.empty() || next_codepoint(value, 0) != value.size()) {
    throw std::runtime_error(
        std::string(wrapper_name) + " field " + std::string(field) +
        " must be a one-character string");
  }
  return value;
}

detail::ReplaceDecoderConfig parse_replace_decoder_config(const json & wrapper) {
  if (!wrapper.contains("pattern") || !wrapper.at("pattern").is_object()) {
    throw std::runtime_error("Replace decoder must contain a pattern object");
  }
  if (!wrapper.contains("content") || !wrapper.at("content").is_string()) {
    throw std::runtime_error("Replace decoder must contain string content");
  }

  detail::ReplaceDecoderConfig config;
  const auto & pattern_wrapper = wrapper.at("pattern");
  if (pattern_wrapper.contains("String") &&
      pattern_wrapper.at("String").is_string()) {
    config.pattern = pattern_wrapper.at("String").get<std::string>();
  } else if (pattern_wrapper.contains("Regex") &&
      pattern_wrapper.at("Regex").is_string()) {
    config.pattern = pattern_wrapper.at("Regex").get<std::string>();
    config.regex_pattern = true;
  } else {
    throw std::runtime_error(
        "Replace decoder pattern must contain string Regex or String");
  }
  config.content = wrapper.at("content").get<std::string>();
  return config;
}

detail::StripDecoderConfig parse_strip_decoder_config(const json & wrapper) {
  detail::StripDecoderConfig config;
  config.content = require_single_codepoint_string(wrapper, "Strip", "content");
  config.start = require_wrapper_uint32(wrapper, "Strip", "start");
  config.stop = require_wrapper_uint32(wrapper, "Strip", "stop");
  return config;
}

detail::CtcDecoderConfig parse_ctc_decoder_config(const json & wrapper) {
  detail::CtcDecoderConfig config;
  config.pad_token =
      optional_wrapper_string(wrapper, "CTC", "pad_token").value_or("<pad>");
  config.word_delimiter_token =
      optional_wrapper_string(wrapper, "CTC", "word_delimiter_token").value_or("|");
  config.cleanup = optional_wrapper_bool(wrapper, "CTC", "cleanup", true);
  return config;
}

detail::DecoderStepConfig parse_decoder_step_config(const json & wrapper);

void append_decoder_steps(
    std::vector<detail::DecoderStepConfig> & steps,
    const json & wrapper) {
  const auto type = require_wrapper_type(wrapper, "decoder", supported_decoder_types());
  if (type == "Sequence") {
    if (!wrapper.contains("decoders") || !wrapper.at("decoders").is_array()) {
      throw std::runtime_error("decoder Sequence must contain a decoders array");
    }
    for (const auto & child : wrapper.at("decoders")) {
      append_decoder_steps(steps, child);
    }
    return;
  }
  steps.push_back(parse_decoder_step_config(wrapper));
}

detail::DecoderStepConfig parse_decoder_step_config(const json & wrapper) {
  const auto type = require_wrapper_type(wrapper, "decoder", supported_decoder_types());
  detail::DecoderStepConfig step;
  if (type == "BPEDecoder") {
    step.kind = detail::DecoderStepKind::Bpe;
    step.bpe.suffix =
        optional_wrapper_string(wrapper, "BPEDecoder", "suffix").value_or("</w>");
  } else if (type == "ByteLevel") {
    step.kind = detail::DecoderStepKind::ByteLevel;
    step.byte_level = parse_byte_level_config(wrapper);
  } else if (type == "WordPiece") {
    step.kind = detail::DecoderStepKind::WordPiece;
    step.wordpiece = parse_wordpiece_decoder_config(wrapper);
  } else if (type == "Metaspace") {
    step.kind = detail::DecoderStepKind::Metaspace;
    step.metaspace = parse_metaspace_config(wrapper);
  } else if (type == "CTC") {
    step.kind = detail::DecoderStepKind::Ctc;
    step.ctc = parse_ctc_decoder_config(wrapper);
  } else if (type == "Replace") {
    step.kind = detail::DecoderStepKind::Replace;
    step.replace = parse_replace_decoder_config(wrapper);
  } else if (type == "Fuse") {
    step.kind = detail::DecoderStepKind::Fuse;
  } else if (type == "Strip") {
    step.kind = detail::DecoderStepKind::Strip;
    step.strip = parse_strip_decoder_config(wrapper);
  } else if (type == "ByteFallback") {
    step.kind = detail::DecoderStepKind::ByteFallback;
  } else {
    throw std::runtime_error("unsupported decoder runtime wrapper type: " + type);
  }
  return step;
}

std::optional<std::vector<detail::DecoderStepConfig>> parse_decoder_steps(
    const json & root) {
  if (!root.contains("decoder") || root.at("decoder").is_null()) {
    return std::nullopt;
  }
  std::vector<detail::DecoderStepConfig> steps;
  append_decoder_steps(steps, root.at("decoder"));
  return steps;
}

detail::SimpleNormalizerConfig parse_simple_normalizer_config(const json & wrapper);

void merge_simple_normalizer_child(
    detail::SimpleNormalizerConfig & config,
    const json & child) {
  if (!child.is_object()) {
    return;
  }
  const auto type = child.value("type", "");
  if (type == "Lowercase") {
    config.enabled = true;
    config.lowercase = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Lowercase, "", ""});
  } else if (type == "NFC") {
    config.enabled = true;
    config.nfc = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Nfc, "", ""});
  } else if (type == "NFD") {
    config.enabled = true;
    config.nfd = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Nfd, "", ""});
  } else if (type == "NFKC") {
    config.enabled = true;
    config.nfkc = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Nfkc, "", ""});
  } else if (type == "NFKD") {
    config.enabled = true;
    config.nfkd = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Nfkd, "", ""});
  } else if (type == "Nmt") {
    config.enabled = true;
    config.nmt = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::Nmt, "", ""});
  } else if (type == "Prepend") {
    if (!child.contains("prepend") || !child.at("prepend").is_string()) {
      throw std::runtime_error("Prepend normalizer must contain string prepend");
    }
    config.enabled = true;
    config.prepend = true;
    config.ops.push_back({
        detail::SimpleNormalizerOpKind::Prepend,
        "",
        child.at("prepend").get<std::string>()});
  } else if (type == "Strip") {
    config.enabled = true;
    config.strip = true;
    detail::SimpleNormalizerOp op{detail::SimpleNormalizerOpKind::Strip, "", ""};
    op.strip_left = optional_wrapper_bool(child, "Strip", "strip_left", true);
    op.strip_right = optional_wrapper_bool(child, "Strip", "strip_right", true);
    config.ops.push_back(std::move(op));
  } else if (type == "StripAccents") {
    config.enabled = true;
    config.strip_accents = true;
    config.ops.push_back({detail::SimpleNormalizerOpKind::StripAccents, "", ""});
  } else if (type == "Precompiled") {
    if (!child.contains("precompiled_charsmap") ||
        !child.at("precompiled_charsmap").is_string()) {
      throw std::runtime_error(
          "Precompiled normalizer must contain string precompiled_charsmap");
    }
    config.enabled = true;
    const auto map_index = config.precompiled_maps.size();
    config.precompiled_maps.push_back(parse_precompiled_charsmap(
        child.at("precompiled_charsmap").get<std::string>()));
    config.ops.push_back({
        detail::SimpleNormalizerOpKind::Precompiled,
        "",
        "",
        map_index});
  } else if (type == "Replace") {
    if (!child.contains("pattern") || !child.at("pattern").is_object()) {
      throw std::runtime_error("Replace normalizer must contain a pattern object");
    }
    if (!child.contains("content") || !child.at("content").is_string()) {
      throw std::runtime_error("Replace normalizer must contain string content");
    }
    const auto & pattern_wrapper = child.at("pattern");
    std::string pattern;
    bool regex_pattern = false;
    if (pattern_wrapper.contains("String") &&
        pattern_wrapper.at("String").is_string()) {
      pattern = pattern_wrapper.at("String").get<std::string>();
    } else if (pattern_wrapper.contains("Regex") &&
        pattern_wrapper.at("Regex").is_string()) {
      pattern = pattern_wrapper.at("Regex").get<std::string>();
      regex_pattern = true;
    } else {
      throw std::runtime_error(
          "Replace normalizer pattern must contain string Regex or String");
    }
    config.enabled = true;
    const auto content = child.at("content").get<std::string>();
    if (!regex_pattern) {
      config.replacements.push_back({
          pattern,
          content});
    }
    detail::SimpleNormalizerOp op{
        detail::SimpleNormalizerOpKind::Replace,
        pattern,
        content};
    op.regex_pattern = regex_pattern;
    config.ops.push_back(std::move(op));
  } else if (type == "Sequence") {
    const auto nested = parse_simple_normalizer_config(child);
    if (nested.enabled) {
      config.enabled = true;
      config.nfc = config.nfc || nested.nfc;
      config.lowercase = config.lowercase || nested.lowercase;
      config.nfd = config.nfd || nested.nfd;
      config.nfkc = config.nfkc || nested.nfkc;
      config.nfkd = config.nfkd || nested.nfkd;
      config.nmt = config.nmt || nested.nmt;
      config.prepend = config.prepend || nested.prepend;
      config.strip = config.strip || nested.strip;
      config.strip_accents = config.strip_accents || nested.strip_accents;
      config.replacements.insert(
          config.replacements.end(),
          nested.replacements.begin(),
          nested.replacements.end());
      const auto precompiled_offset = config.precompiled_maps.size();
      config.precompiled_maps.insert(
          config.precompiled_maps.end(),
          nested.precompiled_maps.begin(),
          nested.precompiled_maps.end());
      for (auto op : nested.ops) {
        if (op.kind == detail::SimpleNormalizerOpKind::Precompiled) {
          op.precompiled_index += precompiled_offset;
        }
        config.ops.push_back(std::move(op));
      }
    }
  }
}

detail::SimpleNormalizerConfig parse_simple_normalizer_config(const json & wrapper) {
  detail::SimpleNormalizerConfig config;
  if (!wrapper.is_object()) {
    return config;
  }
  const auto type = wrapper.value("type", "");
  if (type == "Sequence") {
    if (wrapper.contains("normalizers") && wrapper.at("normalizers").is_array()) {
      for (const auto & child : wrapper.at("normalizers")) {
        merge_simple_normalizer_child(config, child);
      }
    }
  } else {
    merge_simple_normalizer_child(config, wrapper);
  }
  return config;
}

std::optional<detail::SimpleNormalizerConfig> parse_direct_simple_normalizer_config(
    const json & root) {
  if (!root.contains("normalizer") || root.at("normalizer").is_null()) {
    return std::nullopt;
  }
  const auto config = parse_simple_normalizer_config(root.at("normalizer"));
  if (!config.enabled) {
    return std::nullopt;
  }
  return config;
}

std::optional<detail::MetaspaceConfig> parse_direct_metaspace_decoder_config(
    const json & root) {
  if (!root.contains("decoder") || root.at("decoder").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("decoder");
  if (!wrapper.is_object() || wrapper.value("type", "") != "Metaspace") {
    return std::nullopt;
  }
  return parse_metaspace_config(wrapper);
}

std::optional<detail::WordPieceDecoderConfig> parse_direct_wordpiece_decoder_config(
    const json & root) {
  if (!root.contains("decoder") || root.at("decoder").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("decoder");
  if (!wrapper.is_object() || wrapper.value("type", "") != "WordPiece") {
    return std::nullopt;
  }
  return parse_wordpiece_decoder_config(wrapper);
}

std::optional<detail::MetaspaceConfig> parse_sentencepiece_pre_tokenizer_config(
    const json & root,
    bool & whitespace_split) {
  whitespace_split = false;
  if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) {
    return std::nullopt;
  }

  const auto & wrapper = root.at("pre_tokenizer");
  if (!wrapper.is_object()) {
    return std::nullopt;
  }
  if (wrapper.value("type", "") == "Metaspace") {
    return parse_metaspace_config(wrapper);
  }
  if (wrapper.value("type", "") != "Sequence" || !wrapper.contains("pretokenizers") ||
      !wrapper.at("pretokenizers").is_array()) {
    return std::nullopt;
  }

  std::optional<detail::MetaspaceConfig> metaspace;
  for (const auto & child : wrapper.at("pretokenizers")) {
    if (!child.is_object()) {
      continue;
    }
    const auto type = child.value("type", "");
    if (type == "WhitespaceSplit") {
      whitespace_split = true;
    } else if (type == "Metaspace") {
      metaspace = parse_metaspace_config(child);
    }
  }
  return metaspace;
}

bool is_supported_llama_split_regex(const std::string & pattern) {
  return pattern.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") != std::string::npos &&
      pattern.find("\\p{N}{1,3}") != std::string::npos &&
      pattern.find("[^\\s\\p{L}\\p{N}]") != std::string::npos;
}

bool is_supported_llama_stream_escaped_split_regex(const std::string & pattern) {
  return pattern.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") != std::string::npos &&
      pattern.find("\\\\p{N}{1,3}") != std::string::npos &&
      pattern.find("[^\\\\s\\\\p{L}\\\\p{N}]") != std::string::npos;
}

detail::SplitPreTokenizerConfig parse_split_pre_tokenizer_config(const json & wrapper) {
  detail::SplitPreTokenizerConfig config;
  if (!wrapper.contains("pattern") || !wrapper.at("pattern").is_object()) {
    throw std::runtime_error(
        "Split pre_tokenizer runtime requires a pattern object");
  }
  const auto & pattern = wrapper.at("pattern");
  config.enabled = true;
  if (pattern.contains("Regex") && pattern.at("Regex").is_string()) {
    config.regex_pattern = true;
    config.pattern = pattern.at("Regex").get<std::string>();
  } else if (pattern.contains("String") && pattern.at("String").is_string()) {
    config.regex_pattern = false;
    config.pattern = pattern.at("String").get<std::string>();
  } else {
    throw std::runtime_error(
        "Split pre_tokenizer pattern must contain string Regex or String");
  }
  config.behavior = wrapper.value("behavior", "Isolated");
  config.invert = wrapper.value("invert", false);
  if (config.regex_pattern) {
    config.llama_regex = is_supported_llama_split_regex(config.pattern);
    config.llama_stream_escaped_regex =
        is_supported_llama_stream_escaped_split_regex(config.pattern);
  }
  return config;
}

struct PreTokenizerRuntimeConfig {
  std::optional<detail::SplitPreTokenizerConfig> split;
  std::optional<detail::ByteLevelConfig> byte_level;
  std::vector<detail::PreTokenizerStepConfig> steps;
};

detail::PreTokenizerStepConfig split_pre_tokenizer_step(
    detail::SplitPreTokenizerConfig split) {
  detail::PreTokenizerStepConfig step;
  step.kind = detail::PreTokenizerStepKind::Split;
  step.split = std::move(split);
  return step;
}

detail::PreTokenizerStepConfig whitespace_pre_tokenizer_step() {
  detail::PreTokenizerStepConfig step;
  step.kind = detail::PreTokenizerStepKind::Whitespace;
  return step;
}

detail::PreTokenizerStepConfig digits_pre_tokenizer_step(const json & wrapper) {
  detail::PreTokenizerStepConfig step;
  step.kind = detail::PreTokenizerStepKind::Digits;
  step.individual_digits = wrapper.value("individual_digits", false);
  return step;
}

void merge_pre_tokenizer_node(
    PreTokenizerRuntimeConfig & config,
    const json & wrapper) {
  if (!wrapper.is_object()) {
    throw std::runtime_error("pre_tokenizer Sequence children must be objects");
  }
  const auto type = wrapper.value("type", "");
  if (type == "Sequence") {
    if (!wrapper.contains("pretokenizers") || !wrapper.at("pretokenizers").is_array()) {
      throw std::runtime_error(
          "pre_tokenizer Sequence is missing required field pretokenizers");
    }
    for (const auto & child : wrapper.at("pretokenizers")) {
      merge_pre_tokenizer_node(config, child);
    }
    return;
  }
  if (type == "Split") {
    auto split = parse_split_pre_tokenizer_config(wrapper);
    if (!config.split) {
      config.split = split;
    }
    config.steps.push_back(split_pre_tokenizer_step(std::move(split)));
    return;
  }
  if (type == "Whitespace") {
    config.steps.push_back(whitespace_pre_tokenizer_step());
    return;
  }
  if (type == "Digits") {
    config.steps.push_back(digits_pre_tokenizer_step(wrapper));
    return;
  }
  if (type == "ByteLevel") {
    config.byte_level = parse_byte_level_config(wrapper);
  }
}

std::optional<PreTokenizerRuntimeConfig> parse_sequence_pre_tokenizer_config(
    const json & root) {
  if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("pre_tokenizer");
  if (!wrapper.is_object()) {
    return std::nullopt;
  }

  PreTokenizerRuntimeConfig config;
  const auto type = wrapper.value("type", "");
  if (type == "Sequence" || type == "Split" || type == "Whitespace" ||
      type == "Digits" || type == "ByteLevel") {
    merge_pre_tokenizer_node(config, wrapper);
  } else {
    return std::nullopt;
  }
  if (!config.split && !config.byte_level && config.steps.empty()) {
    return std::nullopt;
  }
  return config;
}

detail::TemplateProcessingConfig parse_template_processing_config(const json & wrapper) {
  detail::TemplateProcessingConfig config;
  config.enabled = true;
  if (!wrapper.contains("special_tokens") || !wrapper.at("special_tokens").is_object()) {
    throw std::runtime_error("TemplateProcessing must contain special_tokens object");
  }

  const auto & special_tokens = wrapper.at("special_tokens");
  HashMap<std::string, detail::ProcessingPiece> specials;
  for (const auto & item : special_tokens.items()) {
    if (!item.value().is_object()) {
      throw std::runtime_error("TemplateProcessing special token entries must be objects");
    }
    const auto & entry = item.value();
    if (!entry.contains("ids") || !entry.at("ids").is_array() ||
        !entry.contains("tokens") || !entry.at("tokens").is_array() ||
        entry.at("ids").size() != entry.at("tokens").size()) {
      throw std::runtime_error(
          "TemplateProcessing special token entries must contain matching ids/tokens arrays");
    }
    detail::ProcessingPiece piece;
    piece.kind = detail::ProcessingPieceKind::Special;
    for (std::size_t index = 0; index < entry.at("ids").size(); ++index) {
      if (!is_uint32_json(entry.at("ids").at(index)) ||
          !entry.at("tokens").at(index).is_string()) {
        throw std::runtime_error(
            "TemplateProcessing special token ids/tokens must be uint32/string pairs");
      }
      piece.ids.push_back(entry.at("ids").at(index).get<std::uint32_t>());
      piece.tokens.push_back(entry.at("tokens").at(index).get<std::string>());
    }
    specials.emplace(item.key(), piece);
    if (entry.contains("id") && entry.at("id").is_string()) {
      specials.emplace(entry.at("id").get<std::string>(), piece);
    }
  }

  const auto parse_legacy_special = [&](const std::string & key,
                                        std::string & token,
                                        std::uint32_t & id) {
    const auto special = specials.find(key);
    if (special == specials.end() || special->second.ids.empty() ||
        special->second.tokens.empty()) {
      return;
    }
    token = special->second.tokens.front();
    id = special->second.ids.front();
  };
  parse_legacy_special("[SEP]", config.sep_token, config.sep_id);
  parse_legacy_special("[CLS]", config.cls_token, config.cls_id);

  const auto parse_piece_template =
      [&](const char * field) -> std::vector<detail::ProcessingPiece> {
    if (!wrapper.contains(field) || !wrapper.at(field).is_array()) {
      throw std::runtime_error(
          std::string("TemplateProcessing must contain ") + field + " array");
    }
    std::vector<detail::ProcessingPiece> pieces;
    for (const auto & piece_json : wrapper.at(field)) {
      if (!piece_json.is_object()) {
        throw std::runtime_error("TemplateProcessing pieces must be objects");
      }
      if (piece_json.contains("Sequence")) {
        const auto & sequence = piece_json.at("Sequence");
        if (!sequence.is_object() || !sequence.contains("id") ||
            !sequence.at("id").is_string() || !sequence.contains("type_id") ||
            !is_uint32_json(sequence.at("type_id"))) {
          throw std::runtime_error(
              "TemplateProcessing Sequence pieces must contain id/type_id");
        }
        const auto id = sequence.at("id").get<std::string>();
        const auto type_id = sequence.at("type_id").get<std::uint32_t>();
        if (id == "A" || id == "a") {
          pieces.push_back(sequence_piece(detail::ProcessingPieceKind::SequenceA, type_id));
        } else if (id == "B" || id == "b") {
          pieces.push_back(sequence_piece(detail::ProcessingPieceKind::SequenceB, type_id));
        } else {
          throw std::runtime_error("TemplateProcessing Sequence id must be A or B");
        }
        continue;
      }
      if (piece_json.contains("SpecialToken")) {
        const auto & special = piece_json.at("SpecialToken");
        if (!special.is_object() || !special.contains("id") ||
            !special.at("id").is_string() || !special.contains("type_id") ||
            !is_uint32_json(special.at("type_id"))) {
          throw std::runtime_error(
              "TemplateProcessing SpecialToken pieces must contain id/type_id");
        }
        const auto id = special.at("id").get<std::string>();
        auto found = specials.find(id);
        if (found == specials.end()) {
          throw std::runtime_error(
              "TemplateProcessing SpecialToken references unknown id: " + id);
        }
        auto piece = found->second;
        piece.type_id = special.at("type_id").get<std::uint32_t>();
        pieces.push_back(std::move(piece));
        continue;
      }
      throw std::runtime_error(
          "TemplateProcessing pieces must be Sequence or SpecialToken objects");
    }
    return pieces;
  };

  config.single_template = parse_piece_template("single");
  config.pair_template = parse_piece_template("pair");

  if (config.single_template.empty() || config.pair_template.empty()) {
    throw std::runtime_error("TemplateProcessing templates must not be empty");
  }

  const auto parse_special = [&](const std::string & key,
                                 std::string & token,
                                 std::uint32_t & id) {
    if (!special_tokens.contains(key) || !special_tokens.at(key).is_object()) {
      return;
    }
    const auto & entry = special_tokens.at(key);
    if (entry.contains("tokens") && entry.at("tokens").is_array() &&
        !entry.at("tokens").empty() && entry.at("tokens").at(0).is_string()) {
      token = entry.at("tokens").at(0).get<std::string>();
    }
    if (entry.contains("ids") && entry.at("ids").is_array() &&
        !entry.at("ids").empty() && is_uint32_json(entry.at("ids").at(0))) {
      id = entry.at("ids").at(0).get<std::uint32_t>();
    }
  };
  parse_special("[SEP]", config.sep_token, config.sep_id);
  parse_special("[CLS]", config.cls_token, config.cls_id);
  return config;
}

struct PostProcessorRuntimeConfig {
  std::optional<detail::ByteLevelConfig> byte_level;
  std::optional<detail::BertProcessingConfig> special_processing;
  std::optional<detail::TemplateProcessingConfig> template_processing;
};

void set_special_post_processor(
    PostProcessorRuntimeConfig & config,
    detail::BertProcessingConfig processing) {
  if (config.special_processing || config.template_processing) {
    throw std::runtime_error(
        "post_processor Sequence supports only one special-token processor");
  }
  config.special_processing = std::move(processing);
}

void set_template_post_processor(
    PostProcessorRuntimeConfig & config,
    detail::TemplateProcessingConfig processing) {
  if (config.special_processing || config.template_processing) {
    throw std::runtime_error(
        "post_processor Sequence supports only one special-token processor");
  }
  config.template_processing = std::move(processing);
}

void merge_post_processor_node(
    PostProcessorRuntimeConfig & config,
    const json & wrapper) {
  if (!wrapper.is_object()) {
    throw std::runtime_error("post_processor Sequence children must be objects");
  }
  const auto type = wrapper.value("type", "");
  if (type == "Sequence") {
    if (!wrapper.contains("processors") || !wrapper.at("processors").is_array()) {
      throw std::runtime_error(
          "post_processor Sequence is missing required field processors");
    }
    for (const auto & child : wrapper.at("processors")) {
      merge_post_processor_node(config, child);
    }
    return;
  }
  if (type == "ByteLevel") {
    config.byte_level = parse_byte_level_config(wrapper);
    return;
  }
  if (type == "TemplateProcessing") {
    set_template_post_processor(config, parse_template_processing_config(wrapper));
    return;
  }
  if (type == "BertProcessing") {
    set_special_post_processor(config, parse_bert_processing_config(wrapper));
    return;
  }
  if (type == "RobertaProcessing") {
    auto processing = parse_roberta_processing_config(wrapper);
    if (!config.byte_level) {
      config.byte_level = processing.offset_processor;
    }
    set_special_post_processor(config, std::move(processing));
    return;
  }
  throw std::runtime_error(
      "unsupported post_processor Sequence child type: " + type);
}

std::optional<PostProcessorRuntimeConfig> parse_sequence_post_processor_config(
    const json & root) {
  if (!root.contains("post_processor") || root.at("post_processor").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("post_processor");
  if (!wrapper.is_object() || wrapper.value("type", "") != "Sequence") {
    return std::nullopt;
  }

  PostProcessorRuntimeConfig config;
  merge_post_processor_node(config, wrapper);
  return config;
}

std::optional<detail::TemplateProcessingConfig> parse_direct_template_processing_config(
    const json & root) {
  if (!root.contains("post_processor") || root.at("post_processor").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("post_processor");
  if (!wrapper.is_object() || wrapper.value("type", "") != "TemplateProcessing") {
    return std::nullopt;
  }
  return parse_template_processing_config(wrapper);
}

std::optional<detail::ByteLevelConfig> parse_direct_byte_level_config(
    const json & root,
    const char * slot) {
  if (!root.contains(slot) || root.at(slot).is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at(slot);
  if (!wrapper.is_object() || wrapper.value("type", "") != "ByteLevel") {
    return std::nullopt;
  }
  return parse_byte_level_config(wrapper);
}

std::optional<detail::BertNormalizerConfig> parse_direct_bert_normalizer_config(
    const json & root) {
  if (!root.contains("normalizer") || root.at("normalizer").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("normalizer");
  if (!wrapper.is_object() || wrapper.value("type", "") != "BertNormalizer") {
    return std::nullopt;
  }
  return parse_bert_normalizer_config(wrapper);
}

bool has_direct_bert_pre_tokenizer(const json & root) {
  if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) {
    return false;
  }
  const auto & wrapper = root.at("pre_tokenizer");
  return wrapper.is_object() && wrapper.value("type", "") == "BertPreTokenizer";
}

bool has_direct_pre_tokenizer_type(const json & root, const char * type) {
  if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) {
    return false;
  }
  const auto & wrapper = root.at("pre_tokenizer");
  return wrapper.is_object() && wrapper.value("type", "") == type;
}

std::optional<detail::BertProcessingConfig> parse_direct_bert_processing_config(
    const json & root) {
  if (!root.contains("post_processor") || root.at("post_processor").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("post_processor");
  if (!wrapper.is_object() || wrapper.value("type", "") != "BertProcessing") {
    return std::nullopt;
  }
  return parse_bert_processing_config(wrapper);
}

std::optional<detail::BertProcessingConfig> parse_direct_roberta_processing_config(
    const json & root) {
  if (!root.contains("post_processor") || root.at("post_processor").is_null()) {
    return std::nullopt;
  }
  const auto & wrapper = root.at("post_processor");
  if (!wrapper.is_object() || wrapper.value("type", "") != "RobertaProcessing") {
    return std::nullopt;
  }
  return parse_roberta_processing_config(wrapper);
}

void validate_vocab_object(std::string_view model_type, const json & vocab) {
  if (!vocab.is_object()) {
    throw std::runtime_error(std::string(model_type) + " model vocab must be an object");
  }
  for (const auto & item : vocab.items()) {
    if (!is_uint32_json(item.value())) {
      throw std::runtime_error(
          std::string(model_type) + " model vocab id for token " + item.key() +
          " must be an unsigned 32-bit integer");
    }
  }
}

void validate_bpe_merges(const json & merges) {
  if (!merges.is_array()) {
    throw std::runtime_error("BPE model merges must be an array");
  }
  for (const auto & merge : merges) {
    if (merge.is_string()) {
      continue;
    }
    if (merge.is_array() && merge.size() == 2 && merge.at(0).is_string() &&
        merge.at(1).is_string()) {
      continue;
    }
    throw std::runtime_error("BPE model merge entries must be strings or token pairs");
  }
}

std::pair<std::string, std::string> parse_bpe_merge_pair(const json & merge) {
  if (merge.is_array() && merge.size() == 2 && merge.at(0).is_string() &&
      merge.at(1).is_string()) {
    return {merge.at(0).get<std::string>(), merge.at(1).get<std::string>()};
  }

  if (merge.is_string()) {
    const auto value = merge.get<std::string>();
    const auto separator = value.find(' ');
    if (separator == std::string::npos ||
        value.find(' ', separator + 1) != std::string::npos) {
      throw std::runtime_error("BPE model merge entries must contain one token pair");
    }
    return {value.substr(0, separator), value.substr(separator + 1)};
  }

  throw std::runtime_error("BPE model merge entries must be strings or token pairs");
}

void load_bpe_merges(
    HashMap<std::uint64_t, detail::BpeMerge> & bpe_merges,
    const HashMap<std::string, std::uint32_t> & token_to_id,
    const detail::BpeConfig & config,
    const json & merges) {
  bpe_merges.clear();
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto [left_token, right_token] = parse_bpe_merge_pair(merges.at(rank));
    const auto left = token_to_id.find(left_token);
    const auto right = token_to_id.find(right_token);
    auto merged_right = right_token;
    if (config.continuing_subword_prefix &&
        starts_with(merged_right, *config.continuing_subword_prefix)) {
      merged_right = merged_right.substr(config.continuing_subword_prefix->size());
    }
    const auto merged = left_token + merged_right;
    const auto new_token = token_to_id.find(merged);
    if (left == token_to_id.end() || right == token_to_id.end() ||
        new_token == token_to_id.end()) {
      throw std::runtime_error(
          "BPE model merge token out of vocabulary: " + left_token + " " +
          right_token);
    }
    bpe_merges.emplace(
        bpe_pair_key(left->second, right->second),
        detail::BpeMerge{rank, new_token->second});
  }
}

detail::BpeConfig bpe_config_from_options(const BpeOptions & options) {
  if (options.dropout && (*options.dropout < 0.0 || *options.dropout > 1.0)) {
    throw std::runtime_error("BPE model dropout must be between 0 and 1");
  }

  detail::BpeConfig config;
  config.unk_token = options.unk_token;
  config.continuing_subword_prefix = options.continuing_subword_prefix;
  config.end_of_word_suffix = options.end_of_word_suffix;
  config.dropout = options.dropout;
  config.fuse_unk = options.fuse_unk;
  config.byte_fallback = options.byte_fallback;
  config.ignore_merges = options.ignore_merges;
  return config;
}

json read_bpe_vocab_file(const std::filesystem::path & path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open BPE vocab file: " + path.string());
  }

  json vocab;
  input >> vocab;
  validate_vocab_object("BPE", vocab);
  return vocab;
}

json read_bpe_merges_file(const std::filesystem::path & path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open BPE merges file: " + path.string());
  }

  json merges = json::array();
  std::size_t rank = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (starts_with(line, "#version")) {
      continue;
    }
    ++rank;
    const auto separator = line.find(' ');
    if (separator == std::string::npos ||
        line.find(' ', separator + 1) != std::string::npos) {
      throw std::runtime_error(
          "BPE merges text file invalid at line " + std::to_string(rank));
    }
    merges.push_back(line);
  }
  return merges;
}

void validate_unigram_vocab(const json & vocab) {
  if (!vocab.is_array()) {
    throw std::runtime_error("Unigram model vocab must be an array");
  }
  for (const auto & entry : vocab) {
    if (!entry.is_array() || entry.size() < 2 || !entry.at(0).is_string() ||
        !entry.at(1).is_number()) {
      throw std::runtime_error("Unigram model vocab entries must be [token, score] arrays");
    }
  }
}

std::vector<double> load_unigram_scores(const json & vocab) {
  std::vector<double> scores;
  scores.reserve(vocab.size());
  for (const auto & entry : vocab) {
    scores.push_back(entry.at(1).get<double>());
  }
  return scores;
}

double min_score(const std::vector<double> & scores) {
  if (scores.empty()) {
    return 0.0;
  }
  return *std::min_element(scores.begin(), scores.end());
}

void build_unigram_trie(
    const std::vector<std::string> & id_to_token,
    detail::UnigramConfig & config) {
  config.trie.clear();
  config.trie.push_back(detail::UnigramConfig::TrieNode{});

  const auto vocab_size = std::min(id_to_token.size(), config.scores.size());
  for (std::size_t index = 0; index < vocab_size; ++index) {
    const auto & token = id_to_token[index];
    if (token.empty()) {
      continue;
    }

    std::size_t node_index = 0;
    for (const auto byte_value : token) {
      const auto byte = static_cast<unsigned char>(byte_value);
      auto & children = config.trie[node_index].children;
      const auto child = children.find(byte);
      if (child == children.end()) {
        const auto next_index = config.trie.size();
        children.emplace(byte, next_index);
        config.trie.push_back(detail::UnigramConfig::TrieNode{});
        node_index = next_index;
      } else {
        node_index = child->second;
      }
    }
    config.trie[node_index].token_ids.push_back(static_cast<std::uint32_t>(index));
  }
}

void insert_wordpiece_trie_token(
    std::vector<detail::WordPieceConfig::TrieNode> & trie,
    std::string_view token,
    std::uint32_t id) {
  if (token.empty()) {
    return;
  }
  if (trie.empty()) {
    trie.push_back(detail::WordPieceConfig::TrieNode{});
  }

  std::size_t node_index = 0;
  for (const auto byte_value : token) {
    const auto byte = static_cast<unsigned char>(byte_value);
    auto & children = trie[node_index].children;
    const auto child = children.find(byte);
    if (child == children.end()) {
      const auto next_index = trie.size();
      children.emplace(byte, next_index);
      trie.push_back(detail::WordPieceConfig::TrieNode{});
      node_index = next_index;
    } else {
      node_index = child->second;
    }
  }
  trie[node_index].token_id = id;
}

void build_wordpiece_trie(
    const std::vector<std::string> & id_to_token,
    detail::WordPieceConfig & config) {
  config.initial_trie.clear();
  config.continuation_trie.clear();
  config.initial_trie.push_back(detail::WordPieceConfig::TrieNode{});
  config.continuation_trie.push_back(detail::WordPieceConfig::TrieNode{});

  for (std::size_t index = 0; index < id_to_token.size(); ++index) {
    const auto & token = id_to_token[index];
    if (token.empty()) {
      continue;
    }
    const auto id = static_cast<std::uint32_t>(index);
    insert_wordpiece_trie_token(config.initial_trie, token, id);

    if (config.continuing_subword_prefix.empty()) {
      insert_wordpiece_trie_token(config.continuation_trie, token, id);
    } else if (
        starts_with(token, config.continuing_subword_prefix) &&
        token.size() > config.continuing_subword_prefix.size()) {
      insert_wordpiece_trie_token(
          config.continuation_trie,
          std::string_view(token).substr(config.continuing_subword_prefix.size()),
          id);
    }
  }
}

template <typename Config>
void hydrate_byte_fallback_ids(
    const HashMap<std::string, std::uint32_t> & token_to_id,
    Config & config) {
  config.byte_fallback_ids.fill(std::nullopt);
  for (std::size_t byte = 0; byte < config.byte_fallback_ids.size(); ++byte) {
    const auto token = bpe_byte_fallback_token(static_cast<unsigned char>(byte));
    const auto found = token_to_id.find(token);
    if (found != token_to_id.end()) {
      config.byte_fallback_ids[byte] = found->second;
    }
  }
}

template <typename Impl>
void rebuild_model_runtime_caches(Impl & impl) {
  if (impl.model_type_ == "BPE" && impl.bpe_.unk_token) {
    const auto found = impl.token_to_id_.find(*impl.bpe_.unk_token);
    impl.bpe_.unk_id = found == impl.token_to_id_.end()
        ? std::nullopt
        : std::optional<std::uint32_t>{found->second};
  } else {
    impl.bpe_.unk_id = std::nullopt;
  }
  if (impl.model_type_ == "BPE" && impl.bpe_.byte_fallback) {
    hydrate_byte_fallback_ids(impl.token_to_id_, impl.bpe_);
  } else {
    impl.bpe_.byte_fallback_ids.fill(std::nullopt);
  }

  if (impl.model_type_ == "WordPiece") {
    const auto wordpiece_unk = impl.token_to_id_.find(impl.wordpiece_.unk_token);
    impl.wordpiece_.unk_id = wordpiece_unk == impl.token_to_id_.end()
        ? std::nullopt
        : std::optional<std::uint32_t>{wordpiece_unk->second};
    build_wordpiece_trie(impl.id_to_token_, impl.wordpiece_);
  } else {
    impl.wordpiece_.unk_id = std::nullopt;
    impl.wordpiece_.initial_trie.clear();
    impl.wordpiece_.continuation_trie.clear();
  }

  if (impl.model_type_ == "WordLevel") {
    const auto wordlevel_unk = impl.token_to_id_.find(impl.wordlevel_.unk_token);
    impl.wordlevel_.unk_id = wordlevel_unk == impl.token_to_id_.end()
        ? std::nullopt
        : std::optional<std::uint32_t>{wordlevel_unk->second};
  } else {
    impl.wordlevel_.unk_id = std::nullopt;
  }

  if (impl.model_type_ == "Unigram" && impl.unigram_.byte_fallback) {
    hydrate_byte_fallback_ids(impl.token_to_id_, impl.unigram_);
  } else {
    impl.unigram_.byte_fallback_ids.fill(std::nullopt);
  }
}

std::string model_type_from_json(const json & model) {
  if (!model.is_object()) {
    throw std::runtime_error("model wrapper must be an object");
  }
  if (model.contains("type")) {
    return require_wrapper_type(model, "model", supported_model_types());
  }

  if (model.contains("vocab") && model.at("vocab").is_array()) {
    return "Unigram";
  }
  if (model.contains("merges")) {
    return "BPE";
  }
  if (model.contains("continuing_subword_prefix") ||
      model.contains("max_input_chars_per_word")) {
    return "WordPiece";
  }
  if (model.contains("vocab") && model.at("vocab").is_object()) {
    return "WordLevel";
  }

  throw std::runtime_error("model wrapper must contain a string type");
}

void validate_model(const json & model) {
  const auto type = model_type_from_json(model);

  if (type == "BPE") {
    if (!model.contains("vocab")) {
      throw std::runtime_error("BPE model is missing required vocab");
    }
    validate_vocab_object(type, model.at("vocab"));
    if (!model.contains("merges")) {
      throw std::runtime_error("BPE model is missing required merges");
    }
    validate_bpe_merges(model.at("merges"));
    (void)optional_model_probability_or_null(model, type, "dropout");
    (void)optional_model_string_or_null(model, type, "unk_token");
    (void)optional_model_string_or_null(model, type, "continuing_subword_prefix");
    (void)optional_model_string_or_null(model, type, "end_of_word_suffix");
    (void)optional_model_bool(model, type, "fuse_unk", false);
    (void)optional_model_bool(model, type, "byte_fallback", false);
    (void)optional_model_bool(model, type, "ignore_merges", false);
    return;
  }

  if (type == "WordPiece") {
    if (!model.contains("vocab")) {
      throw std::runtime_error("WordPiece model is missing required vocab");
    }
    validate_vocab_object(type, model.at("vocab"));
    if (model.contains("unk_token") && !model.at("unk_token").is_string()) {
      throw std::runtime_error("WordPiece model unk_token must be a string");
    }
    if (model.contains("continuing_subword_prefix") &&
        !model.at("continuing_subword_prefix").is_string()) {
      throw std::runtime_error(
          "WordPiece model continuing_subword_prefix must be a string");
    }
    if (model.contains("max_input_chars_per_word")) {
      (void)require_uint32(model, "max_input_chars_per_word");
    }
    return;
  }

  if (type == "WordLevel") {
    if (!model.contains("vocab")) {
      throw std::runtime_error("WordLevel model is missing required vocab");
    }
    validate_vocab_object(type, model.at("vocab"));
    if (model.contains("unk_token") && !model.at("unk_token").is_string()) {
      throw std::runtime_error("WordLevel model unk_token must be a string");
    }
    return;
  }

  if (type == "Unigram") {
    if (!model.contains("vocab")) {
      throw std::runtime_error("Unigram model is missing required vocab");
    }
    validate_unigram_vocab(model.at("vocab"));
    if (model.contains("unk_id") && !model.at("unk_id").is_null()) {
      const auto unk_id = require_uint32(model, "unk_id");
      if (unk_id >= model.at("vocab").size()) {
        throw std::runtime_error("Unigram model unk_id must be smaller than vocab size");
      }
    }
    (void)optional_model_bool(model, type, "byte_fallback", false);
    if (model.contains("fuse_unk")) {
      (void)optional_model_bool(model, type, "fuse_unk", true);
    }
  }
}

void validate_optional_wrapper(
    const json & root,
    const char * slot,
    const std::unordered_set<std::string> & supported);

void validate_sequence_children(
    const json & wrapper,
    std::string_view slot,
    std::string_view child_field,
    const std::unordered_set<std::string> & supported) {
  if (!wrapper.contains(child_field)) {
    throw std::runtime_error(
        std::string(slot) + " Sequence is missing required field " +
        std::string(child_field));
  }
  const auto & children = wrapper.at(child_field);
  if (!children.is_array()) {
    throw std::runtime_error(
        std::string(slot) + " Sequence field " + std::string(child_field) +
        " must be an array");
  }
  for (const auto & child : children) {
    require_wrapper_type(child, slot, supported);
    if (child.value("type", "") == "Sequence") {
      validate_sequence_children(child, slot, child_field, supported);
    }
  }
}

void validate_optional_wrapper(
    const json & root,
    const char * slot,
    const std::unordered_set<std::string> & supported) {
  if (!root.contains(slot) || root.at(slot).is_null()) {
    return;
  }

  const auto & wrapper = root.at(slot);
  const auto type = require_wrapper_type(wrapper, slot, supported);
  if (type != "Sequence") {
    return;
  }

  if (std::string_view(slot) == "normalizer") {
    validate_sequence_children(wrapper, slot, "normalizers", supported);
  } else if (std::string_view(slot) == "pre_tokenizer") {
    validate_sequence_children(wrapper, slot, "pretokenizers", supported);
  } else if (std::string_view(slot) == "post_processor") {
    validate_sequence_children(wrapper, slot, "processors", supported);
  } else if (std::string_view(slot) == "decoder") {
    validate_sequence_children(wrapper, slot, "decoders", supported);
  }
}

void load_special_piece_vocab(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const detail::ProcessingPiece & piece) {
  if (piece.kind != detail::ProcessingPieceKind::Special) {
    return;
  }
  if (piece.ids.size() != piece.tokens.size()) {
    throw std::runtime_error("post-processor special token ids/tokens size mismatch");
  }
  for (std::size_t index = 0; index < piece.ids.size(); ++index) {
    load_vocab_entry(
        id_to_token,
        token_to_id,
        special_id,
        piece.tokens[index],
        piece.ids[index],
        true);
  }
}

void load_processing_template_vocab(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const std::vector<detail::ProcessingPiece> & pieces) {
  for (const auto & piece : pieces) {
    load_special_piece_vocab(id_to_token, token_to_id, special_id, piece);
  }
}

void load_special_processing_vocab(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const detail::BertProcessingConfig & config) {
  load_processing_template_vocab(
      id_to_token,
      token_to_id,
      special_id,
      config.single_template);
  load_processing_template_vocab(
      id_to_token,
      token_to_id,
      special_id,
      config.pair_template);
}

void load_template_processing_vocab(
    std::vector<std::string> & id_to_token,
    HashMap<std::string, std::uint32_t> & token_to_id,
    std::vector<bool> & special_id,
    const detail::TemplateProcessingConfig & config) {
  load_processing_template_vocab(
      id_to_token,
      token_to_id,
      special_id,
      config.single_template);
  load_processing_template_vocab(
      id_to_token,
      token_to_id,
      special_id,
      config.pair_template);
}

}  // namespace

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}

Tokenizer::~Tokenizer() = default;

Tokenizer::Tokenizer(const Tokenizer & other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr) {}

Tokenizer & Tokenizer::operator=(const Tokenizer & other) {
  if (this != &other) {
    if (other.impl_) {
      if (!impl_) {
        impl_ = std::make_unique<Impl>(*other.impl_);
      } else {
        *impl_ = *other.impl_;
      }
    } else {
      impl_.reset();
    }
  }
  return *this;
}

Tokenizer::Tokenizer(Tokenizer && other) noexcept = default;

Tokenizer & Tokenizer::operator=(Tokenizer && other) noexcept = default;

DecodeStream::DecodeStream(
    const Tokenizer & tokenizer,
    bool skip_special_tokens)
    : tokenizer_(&tokenizer),
      skip_special_tokens_(skip_special_tokens) {}

bool DecodeStream::has_pending() const {
  return !pending_.empty() || !pending_byte_fallback_.empty();
}

std::optional<std::string> DecodeStream::step(std::uint32_t id) {
  if (tokenizer_ == nullptr) {
    return std::nullopt;
  }

  const auto has_decoder_step = [this](detail::DecoderStepKind kind) {
    return std::any_of(
        tokenizer_->impl_->decoder_steps_.begin(),
        tokenizer_->impl_->decoder_steps_.end(),
        [kind](const detail::DecoderStepConfig & step) {
          return step.kind == kind;
        });
  };

  if (has_decoder_step(detail::DecoderStepKind::ByteFallback)) {
    if (id >= tokenizer_->impl_->id_to_token_.size()) {
      return std::nullopt;
    }
    if (skip_special_tokens_ && id < tokenizer_->impl_->special_id_.size() &&
        tokenizer_->impl_->special_id_[id]) {
      return std::nullopt;
    }

    const auto & token = tokenizer_->impl_->id_to_token_[id];
    if (const auto byte = byte_fallback_token_value(token)) {
      pending_byte_fallback_.push_back(static_cast<char>(*byte));
      const auto prefix_size = valid_utf8_prefix_size(pending_byte_fallback_);
      if (prefix_size != pending_byte_fallback_.size()) {
        return std::nullopt;
      }
      auto output = pending_byte_fallback_;
      pending_byte_fallback_.clear();
      return output.empty() ? std::nullopt : std::optional<std::string>(output);
    }

    std::string prefix;
    if (!pending_byte_fallback_.empty()) {
      std::vector<std::string> flushed;
      flush_byte_fallback_run(flushed, pending_byte_fallback_);
      prefix = join_decoded_tokens(flushed);
    }

    const auto piece = tokenizer_->decode({id}, skip_special_tokens_);
    if (prefix.empty() && piece.empty()) {
      return std::nullopt;
    }
    return prefix + piece;
  }

  if (tokenizer_->impl_->metaspace_decoder_.enabled ||
      has_decoder_step(detail::DecoderStepKind::Metaspace)) {
    ids_.push_back(id);
    const auto decoded = tokenizer_->decode(ids_, skip_special_tokens_);
    if (decoded.empty()) {
      return std::nullopt;
    }
    if (!starts_with(decoded, emitted_)) {
      emitted_ = decoded;
      return emitted_;
    }
    if (decoded.size() <= emitted_.size()) {
      return std::nullopt;
    }
    auto output = decoded.substr(emitted_.size());
    emitted_ = decoded;
    return output.empty() ? std::nullopt : std::optional<std::string>(output);
  }

  const auto piece = tokenizer_->decode({id}, skip_special_tokens_);
  if (piece.empty()) {
    return std::nullopt;
  }

  pending_.append(piece);
  const auto prefix_size = valid_utf8_prefix_size(pending_);
  if (prefix_size == 0) {
    return std::nullopt;
  }

  auto output = pending_.substr(0, prefix_size);
  pending_.erase(0, prefix_size);
  if (output.empty()) {
    return std::nullopt;
  }
  return output;
}

Tokenizer Tokenizer::from_file(const std::filesystem::path & path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open tokenizer file: " + path.string());
  }

  json root;
  input >> root;
  if (!root.is_object()) {
    throw std::runtime_error("tokenizer JSON root must be an object");
  }

  if (root.contains("version") &&
      (!root.at("version").is_string() || root.at("version").get<std::string>() != "1.0")) {
    throw std::runtime_error("unsupported tokenizer JSON version");
  }

  if (!root.contains("model")) {
    throw std::runtime_error("tokenizer JSON is missing required model wrapper");
  }
  const auto & model = root.at("model");
  validate_model(model);
  validate_optional_wrapper(root, "normalizer", supported_normalizer_types());
  validate_optional_wrapper(root, "pre_tokenizer", supported_pre_tokenizer_types());
  validate_optional_wrapper(root, "post_processor", supported_post_processor_types());
  validate_optional_wrapper(root, "decoder", supported_decoder_types());

  Tokenizer tokenizer;
  tokenizer.impl_->model_type_ = model_type_from_json(model);
  if (model.contains("vocab")) {
    const auto & vocab = model.at("vocab");
    if (vocab.is_object()) {
      load_vocab_object(
          tokenizer.impl_->id_to_token_,
          tokenizer.impl_->token_to_id_,
          tokenizer.impl_->special_id_,
          vocab);
    } else if (vocab.is_array()) {
      load_vocab_array(
          tokenizer.impl_->id_to_token_,
          tokenizer.impl_->token_to_id_,
          tokenizer.impl_->special_id_,
          vocab);
    }
  }
  rebuild_token_to_id(tokenizer.impl_->id_to_token_, tokenizer.impl_->token_to_id_);
  if (tokenizer.impl_->model_type_ == "BPE") {
    tokenizer.impl_->bpe_.dropout =
        optional_model_probability_or_null(model, "BPE", "dropout");
    tokenizer.impl_->bpe_.unk_token = optional_model_string_or_null(model, "BPE", "unk_token");
    tokenizer.impl_->bpe_.continuing_subword_prefix =
        optional_model_string_or_null(model, "BPE", "continuing_subword_prefix");
    tokenizer.impl_->bpe_.end_of_word_suffix =
        optional_model_string_or_null(model, "BPE", "end_of_word_suffix");
    tokenizer.impl_->bpe_.fuse_unk = optional_model_bool(model, "BPE", "fuse_unk", false);
    tokenizer.impl_->bpe_.byte_fallback =
        optional_model_bool(model, "BPE", "byte_fallback", false);
    tokenizer.impl_->bpe_.ignore_merges =
        optional_model_bool(model, "BPE", "ignore_merges", false);
    load_bpe_merges(
        tokenizer.impl_->bpe_merges_,
        tokenizer.impl_->token_to_id_,
        tokenizer.impl_->bpe_,
        model.at("merges"));
  }
  if (tokenizer.impl_->model_type_ == "WordPiece") {
    tokenizer.impl_->wordpiece_.unk_token = model.value("unk_token", "[UNK]");
    tokenizer.impl_->wordpiece_.continuing_subword_prefix =
        model.value("continuing_subword_prefix", "##");
    tokenizer.impl_->wordpiece_.max_input_chars_per_word =
        model.value("max_input_chars_per_word", 100U);
  }
  if (tokenizer.impl_->model_type_ == "WordLevel") {
    tokenizer.impl_->wordlevel_.unk_token = model.value("unk_token", "<unk>");
  }
  if (tokenizer.impl_->model_type_ == "Unigram") {
    if (model.contains("unk_id") && !model.at("unk_id").is_null()) {
      tokenizer.impl_->unigram_.unk_id = model.at("unk_id").get<std::uint32_t>();
    }
    tokenizer.impl_->unigram_.byte_fallback =
        optional_model_bool(model, "Unigram", "byte_fallback", false);
    tokenizer.impl_->unigram_.fuse_unk =
        optional_model_bool(model, "Unigram", "fuse_unk", true);
    tokenizer.impl_->unigram_.scores = load_unigram_scores(model.at("vocab"));
    tokenizer.impl_->unigram_.min_score = min_score(tokenizer.impl_->unigram_.scores);
    build_unigram_trie(tokenizer.impl_->id_to_token_, tokenizer.impl_->unigram_);
  }

  if (auto config = parse_direct_bert_normalizer_config(root)) {
    tokenizer.impl_->bert_normalizer_ = *config;
  }
  if (auto config = parse_direct_byte_level_config(root, "normalizer")) {
    tokenizer.impl_->byte_level_normalizer_ = config->enabled;
  }
  if (auto config = parse_direct_simple_normalizer_config(root)) {
    tokenizer.impl_->simple_normalizer_ = *config;
  }
  tokenizer.impl_->bert_pre_tokenizer_ = has_direct_bert_pre_tokenizer(root);
  tokenizer.impl_->whitespace_pre_tokenizer_ =
      has_direct_pre_tokenizer_type(root, "Whitespace");
  const auto direct_whitespace_split_pre_tokenizer =
      has_direct_pre_tokenizer_type(root, "WhitespaceSplit");
  if (auto config = parse_sentencepiece_pre_tokenizer_config(
          root,
          tokenizer.impl_->whitespace_split_pre_tokenizer_)) {
    tokenizer.impl_->metaspace_pre_tokenizer_ = *config;
  }
  tokenizer.impl_->whitespace_split_pre_tokenizer_ =
      tokenizer.impl_->whitespace_split_pre_tokenizer_ ||
      direct_whitespace_split_pre_tokenizer;
  if (auto config = parse_sequence_pre_tokenizer_config(root)) {
    if (config->split) {
      tokenizer.impl_->split_pre_tokenizer_ = *config->split;
    }
    if (config->byte_level) {
      tokenizer.impl_->byte_level_pre_tokenizer_ = *config->byte_level;
    }
    tokenizer.impl_->pre_tokenizer_steps_ = std::move(config->steps);
  }
  if (auto config = parse_direct_bert_processing_config(root)) {
    tokenizer.impl_->bert_processing_ = *config;
    load_special_processing_vocab(
        tokenizer.impl_->id_to_token_,
        tokenizer.impl_->token_to_id_,
        tokenizer.impl_->special_id_,
        tokenizer.impl_->bert_processing_);
  }
  if (auto config = parse_direct_roberta_processing_config(root)) {
    tokenizer.impl_->bert_processing_ = *config;
    load_special_processing_vocab(
        tokenizer.impl_->id_to_token_,
        tokenizer.impl_->token_to_id_,
        tokenizer.impl_->special_id_,
        tokenizer.impl_->bert_processing_);
    tokenizer.impl_->byte_level_post_processor_ =
        tokenizer.impl_->bert_processing_.offset_processor;
  }
  if (auto config = parse_direct_template_processing_config(root)) {
    tokenizer.impl_->template_processing_ = *config;
    load_template_processing_vocab(
        tokenizer.impl_->id_to_token_,
        tokenizer.impl_->token_to_id_,
        tokenizer.impl_->special_id_,
        tokenizer.impl_->template_processing_);
  }
  if (auto config = parse_sequence_post_processor_config(root)) {
    if (config->byte_level) {
      tokenizer.impl_->byte_level_post_processor_ = *config->byte_level;
    }
    if (config->special_processing) {
      tokenizer.impl_->bert_processing_ = *config->special_processing;
      load_special_processing_vocab(
          tokenizer.impl_->id_to_token_,
          tokenizer.impl_->token_to_id_,
          tokenizer.impl_->special_id_,
          tokenizer.impl_->bert_processing_);
    }
    if (config->template_processing) {
      tokenizer.impl_->template_processing_ = *config->template_processing;
      load_template_processing_vocab(
          tokenizer.impl_->id_to_token_,
          tokenizer.impl_->token_to_id_,
          tokenizer.impl_->special_id_,
          tokenizer.impl_->template_processing_);
    }
  }
  if (auto config = parse_direct_byte_level_config(root, "pre_tokenizer")) {
    tokenizer.impl_->byte_level_pre_tokenizer_ = *config;
  }
  if (auto config = parse_direct_byte_level_config(root, "post_processor")) {
    tokenizer.impl_->byte_level_post_processor_ = *config;
  }
  if (auto config = parse_direct_byte_level_config(root, "decoder")) {
    tokenizer.impl_->byte_level_decoder_ = *config;
  }
  if (auto config = parse_direct_metaspace_decoder_config(root)) {
    tokenizer.impl_->metaspace_decoder_ = *config;
  }
  if (auto config = parse_direct_wordpiece_decoder_config(root)) {
    tokenizer.impl_->wordpiece_decoder_ = *config;
  }
  if (auto steps = parse_decoder_steps(root)) {
    tokenizer.impl_->decoder_steps_ = *steps;
  }
  if (auto config = parse_truncation_config(root)) {
    tokenizer.impl_->truncation_ = *config;
  }
  if (auto config = parse_padding_config(root)) {
    tokenizer.impl_->padding_ = *config;
    load_vocab_entry(
        tokenizer.impl_->id_to_token_,
        tokenizer.impl_->token_to_id_,
        tokenizer.impl_->special_id_,
        tokenizer.impl_->padding_.pad_token,
        tokenizer.impl_->padding_.pad_id,
        false);
  }

  if (root.contains("added_tokens")) {
    if (!root.at("added_tokens").is_array()) {
      throw std::runtime_error("tokenizer JSON added_tokens must be an array");
    }
    for (const auto & token : root.at("added_tokens")) {
      auto added_token = parse_added_token(token);
      if (added_token.content.empty()) {
        continue;
      }

      const auto found_id = tokenizer.impl_->token_to_id_.find(added_token.content);
      added_token.id = found_id == tokenizer.impl_->token_to_id_.end()
          ? static_cast<std::uint32_t>(tokenizer.impl_->id_to_token_.size())
          : found_id->second;
      load_vocab_entry(
          tokenizer.impl_->id_to_token_,
          tokenizer.impl_->token_to_id_,
          tokenizer.impl_->special_id_,
          added_token.content,
          added_token.id,
          added_token.special);
      tokenizer.impl_->added_tokens_.push_back(std::move(added_token));
    }
    rebuild_added_token_matcher(
        tokenizer.impl_->added_tokens_,
        tokenizer.impl_->added_token_matcher_);
  }

  rebuild_token_to_id(tokenizer.impl_->id_to_token_, tokenizer.impl_->token_to_id_);
  rebuild_model_runtime_caches(*tokenizer.impl_);
  return tokenizer;
}

Tokenizer Tokenizer::from_bpe_files(
    const std::filesystem::path & vocab_path,
    const std::filesystem::path & merges_path,
    const BpeOptions & options) {
  const auto vocab = read_bpe_vocab_file(vocab_path);
  const auto merges = read_bpe_merges_file(merges_path);
  validate_bpe_merges(merges);

  Tokenizer tokenizer;
  tokenizer.impl_->model_type_ = "BPE";
  tokenizer.impl_->bpe_ = bpe_config_from_options(options);
  load_vocab_object(
      tokenizer.impl_->id_to_token_,
      tokenizer.impl_->token_to_id_,
      tokenizer.impl_->special_id_,
      vocab);
  rebuild_token_to_id(tokenizer.impl_->id_to_token_, tokenizer.impl_->token_to_id_);
  load_bpe_merges(
      tokenizer.impl_->bpe_merges_,
      tokenizer.impl_->token_to_id_,
      tokenizer.impl_->bpe_,
      merges);
  rebuild_token_to_id(tokenizer.impl_->id_to_token_, tokenizer.impl_->token_to_id_);
  rebuild_model_runtime_caches(*tokenizer.impl_);
  return tokenizer;
}

std::string byte_level_normalize_text(std::string_view text) {
  const auto split = InputSplit{
      false,
      0,
      std::string(text),
      Offset{0, text.size()}};
  return byte_level_normalize_split(split).text;
}

std::size_t Tokenizer::add_tokens(const std::vector<AddedToken> & tokens) {
  std::size_t added = 0;
  for (const auto & token : tokens) {
    if (token.content.empty()) {
      continue;
    }

    detail::AddedToken added_token;
    added_token.content = token.content;
    added_token.single_word = token.single_word;
    added_token.lstrip = token.lstrip;
    added_token.rstrip = token.rstrip;
    added_token.normalized = token.normalized;
    added_token.special = token.special;
    if (token.normalized && impl_->byte_level_normalizer_) {
      const auto normalized = byte_level_normalize_text(token.content);
      if (normalized != token.content) {
        added_token.normalized_content = normalized;
      }
    }

    bool duplicate = false;
    for (const auto & existing : impl_->added_tokens_) {
      if (existing.content == added_token.content &&
          existing.single_word == added_token.single_word &&
          existing.lstrip == added_token.lstrip &&
          existing.rstrip == added_token.rstrip &&
          existing.normalized == added_token.normalized &&
          existing.special == added_token.special &&
          existing.normalized_content == added_token.normalized_content) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }

    const auto found_id = impl_->token_to_id_.find(added_token.content);
    const bool has_existing_id = found_id != impl_->token_to_id_.end();
    added_token.id = has_existing_id
        ? found_id->second
        : static_cast<std::uint32_t>(impl_->id_to_token_.size());

    if (has_existing_id) {
      ensure_id(impl_->id_to_token_, impl_->special_id_, added_token.id);
      impl_->special_id_[added_token.id] = impl_->special_id_[added_token.id] || added_token.special;
    } else {
      const auto & stored_token = added_token.normalized_content.empty()
          ? added_token.content
          : added_token.normalized_content;
      load_vocab_entry(
          impl_->id_to_token_,
          impl_->token_to_id_,
          impl_->special_id_,
          stored_token,
          added_token.id,
          added_token.special);
    }
    impl_->added_tokens_.push_back(std::move(added_token));
    ++added;
  }
  if (added > 0) {
    rebuild_token_to_id(impl_->id_to_token_, impl_->token_to_id_);
    impl_->bpe_cache_id_ = allocate_private_cache_id();
    impl_->unigram_cache_id_ = allocate_private_cache_id();
    rebuild_model_runtime_caches(*impl_);
    rebuild_added_token_matcher(impl_->added_tokens_, impl_->added_token_matcher_);
  }
  return added;
}

void Tokenizer::with_byte_level_normalizer() {
  impl_->byte_level_normalizer_ = true;
  impl_->simple_normalizer_ = detail::SimpleNormalizerConfig{};
  impl_->bert_normalizer_ = detail::BertNormalizerConfig{};
}

void Tokenizer::with_split_pre_tokenizer(const std::string & regex_pattern) {
  detail::SplitPreTokenizerConfig config;
  config.enabled = true;
  config.pattern = regex_pattern;
  config.behavior = "Isolated";
  config.invert = false;
  config.llama_regex = is_supported_llama_split_regex(regex_pattern);
  config.llama_stream_escaped_regex =
      is_supported_llama_stream_escaped_split_regex(regex_pattern);
  if (!config.llama_regex && !config.llama_stream_escaped_regex) {
    throw std::runtime_error("unsupported Split pre-tokenizer runtime configuration");
  }

  impl_->split_pre_tokenizer_ = std::move(config);
  impl_->pre_tokenizer_steps_.clear();
  impl_->pre_tokenizer_steps_.push_back(split_pre_tokenizer_step(impl_->split_pre_tokenizer_));
  impl_->byte_level_pre_tokenizer_ = detail::ByteLevelConfig{};
  impl_->bert_pre_tokenizer_ = false;
  impl_->whitespace_pre_tokenizer_ = false;
  impl_->whitespace_split_pre_tokenizer_ = false;
  impl_->metaspace_pre_tokenizer_ = detail::MetaspaceConfig{};
}

void Tokenizer::with_wordpiece_decoder(std::string prefix, bool cleanup) {
  impl_->decoder_steps_.clear();
  impl_->byte_level_decoder_ = detail::ByteLevelConfig{};
  impl_->metaspace_decoder_ = detail::MetaspaceConfig{};
  impl_->wordpiece_decoder_.enabled = true;
  impl_->wordpiece_decoder_.prefix = std::move(prefix);
  impl_->wordpiece_decoder_.cleanup = cleanup;
}

Encoding Tokenizer::encode(const std::string & text, bool add_special_tokens) const {
  return encode_text(text, add_special_tokens, true, true);
}

Encoding Tokenizer::encode_text(
    const std::string & text,
    bool add_special_tokens,
    bool apply_truncation,
    bool apply_padding_config) const {
  Encoding encoding;
  const auto splits =
      split_on_added_tokens(text, impl_->added_tokens_, impl_->added_token_matcher_.get());

  std::uint32_t word_id = 0;
  for (const auto & split : splits) {
    encode_input_split(
        encoding,
        split,
        impl_->id_to_token_,
        impl_->special_id_,
        impl_->model_type_,
        impl_->token_to_id_,
        impl_->bpe_merges_,
        impl_->bpe_,
        impl_->bpe_cache_id_,
        impl_->wordpiece_,
        impl_->wordlevel_,
        impl_->unigram_,
        impl_->unigram_cache_id_,
        impl_->byte_level_normalizer_,
        impl_->simple_normalizer_,
        impl_->bert_normalizer_,
        impl_->bert_pre_tokenizer_,
        impl_->whitespace_pre_tokenizer_,
        impl_->whitespace_split_pre_tokenizer_,
        impl_->metaspace_pre_tokenizer_,
        impl_->split_pre_tokenizer_,
        impl_->pre_tokenizer_steps_,
        impl_->byte_level_pre_tokenizer_,
        std::nullopt,
        word_id);
  }

  process_byte_level_offsets(encoding, impl_->byte_level_post_processor_);
  const auto special_processing =
      effective_special_processing(impl_->bert_processing_, impl_->template_processing_);
  if (apply_truncation) {
    truncate_single_before_processing(
        encoding,
        impl_->truncation_,
        special_processing,
        add_special_tokens);
  }
  auto processed = apply_bert_processing_single(
      std::move(encoding),
      special_processing,
      add_special_tokens);
  if (apply_padding_config) {
    apply_padding(processed, impl_->padding_);
  }
  return processed;
}

Encoding Tokenizer::encode(
    const std::vector<std::string> & pre_tokenized,
    bool add_special_tokens) const {
  return encode_pre_tokenized_words(pre_tokenized, add_special_tokens, true, true);
}

Encoding Tokenizer::encode_char_offsets(
    const std::string & text,
    bool add_special_tokens) const {
  auto encoding = encode_text(text, add_special_tokens, true, true);
  convert_offsets_to_char_offsets(encoding, text);
  return encoding;
}

Encoding Tokenizer::encode_char_offsets(
    const std::vector<std::string> & pre_tokenized,
    bool add_special_tokens) const {
  auto encoding =
      encode_pre_tokenized_words(pre_tokenized, add_special_tokens, true, true);
  convert_pre_tokenized_offsets_to_char_offsets(encoding, pre_tokenized);
  return encoding;
}

Encoding Tokenizer::encode_pre_tokenized_words(
    const std::vector<std::string> & pre_tokenized,
    bool add_special_tokens,
    bool apply_truncation,
    bool apply_padding_config) const {
  Encoding encoding;

  std::uint32_t next_word_id = 0;
  for (std::size_t index = 0; index < pre_tokenized.size(); ++index) {
    if (index > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("pre-tokenized input has too many words");
    }
    const auto word_id = static_cast<std::uint32_t>(index);
    const auto splits = split_on_added_tokens(
        pre_tokenized[index],
        impl_->added_tokens_,
        impl_->added_token_matcher_.get());
    for (const auto & split : splits) {
      encode_input_split(
          encoding,
          split,
          impl_->id_to_token_,
          impl_->special_id_,
          impl_->model_type_,
          impl_->token_to_id_,
          impl_->bpe_merges_,
          impl_->bpe_,
          impl_->bpe_cache_id_,
          impl_->wordpiece_,
          impl_->wordlevel_,
          impl_->unigram_,
          impl_->unigram_cache_id_,
          impl_->byte_level_normalizer_,
          impl_->simple_normalizer_,
          impl_->bert_normalizer_,
          impl_->bert_pre_tokenizer_,
          impl_->whitespace_pre_tokenizer_,
          impl_->whitespace_split_pre_tokenizer_,
          impl_->metaspace_pre_tokenizer_,
          impl_->split_pre_tokenizer_,
          impl_->pre_tokenizer_steps_,
          impl_->byte_level_pre_tokenizer_,
          word_id,
          next_word_id);
    }
  }

  process_byte_level_offsets(encoding, impl_->byte_level_post_processor_);
  const auto special_processing =
      effective_special_processing(impl_->bert_processing_, impl_->template_processing_);
  if (apply_truncation) {
    truncate_single_before_processing(
        encoding,
        impl_->truncation_,
        special_processing,
        add_special_tokens);
  }
  auto processed = apply_bert_processing_single(
      std::move(encoding),
      special_processing,
      add_special_tokens);
  if (apply_padding_config) {
    apply_padding(processed, impl_->padding_);
  }
  return processed;
}

Encoding Tokenizer::encode_pair(
    const std::string & text_a,
    const std::string & text_b,
    bool add_special_tokens) const {
  return encode_pair_text(text_a, text_b, add_special_tokens, true);
}

Encoding Tokenizer::encode_pair(
    const std::vector<std::string> & pre_tokenized_a,
    const std::vector<std::string> & pre_tokenized_b,
    bool add_special_tokens) const {
  return encode_pair_pre_tokenized_words(
      pre_tokenized_a,
      pre_tokenized_b,
      add_special_tokens,
      true);
}

Encoding Tokenizer::encode_pair_char_offsets(
    const std::string & text_a,
    const std::string & text_b,
    bool add_special_tokens) const {
  auto encoding = encode_pair_text(text_a, text_b, add_special_tokens, true);
  convert_pair_offsets_to_char_offsets(encoding, text_a, text_b);
  return encoding;
}

Encoding Tokenizer::encode_pair_char_offsets(
    const std::vector<std::string> & pre_tokenized_a,
    const std::vector<std::string> & pre_tokenized_b,
    bool add_special_tokens) const {
  auto encoding = encode_pair_pre_tokenized_words(
      pre_tokenized_a,
      pre_tokenized_b,
      add_special_tokens,
      true);
  convert_pre_tokenized_pair_offsets_to_char_offsets(
      encoding,
      pre_tokenized_a,
      pre_tokenized_b);
  return encoding;
}

Encoding Tokenizer::encode_pair_text(
    const std::string & text_a,
    const std::string & text_b,
    bool add_special_tokens,
    bool apply_padding_config) const {
  Encoding first = encode_text(text_a, false, false, false);
  Encoding second = encode_text(text_b, false, false, false);
  const auto special_processing =
      effective_special_processing(impl_->bert_processing_, impl_->template_processing_);
  truncate_pair_before_processing(
      first,
      second,
      impl_->truncation_,
      special_processing,
      add_special_tokens);
  auto processed = apply_bert_processing_pair(
      std::move(first),
      std::move(second),
      special_processing,
      add_special_tokens);
  if (apply_padding_config) {
    apply_padding(processed, impl_->padding_);
  }
  return processed;
}

Encoding Tokenizer::encode_pair_pre_tokenized_words(
    const std::vector<std::string> & pre_tokenized_a,
    const std::vector<std::string> & pre_tokenized_b,
    bool add_special_tokens,
    bool apply_padding_config) const {
  Encoding first = encode_pre_tokenized_words(
      pre_tokenized_a,
      false,
      false,
      false);
  Encoding second = encode_pre_tokenized_words(
      pre_tokenized_b,
      false,
      false,
      false);
  const auto special_processing =
      effective_special_processing(impl_->bert_processing_, impl_->template_processing_);
  truncate_pair_before_processing(
      first,
      second,
      impl_->truncation_,
      special_processing,
      add_special_tokens);
  auto processed = apply_bert_processing_pair(
      std::move(first),
      std::move(second),
      special_processing,
      add_special_tokens);
  if (apply_padding_config) {
    apply_padding(processed, impl_->padding_);
  }
  return processed;
}

std::vector<Encoding> Tokenizer::encode_batch(
    const std::vector<std::string> & texts,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      texts.size(),
      total_text_bytes(texts),
      [this, &texts, add_special_tokens](std::size_t index) {
        return encode_text(texts[index], add_special_tokens, true, false);
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch(
    const std::vector<std::vector<std::string>> & pre_tokenized_texts,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pre_tokenized_texts.size(),
      total_text_bytes(pre_tokenized_texts),
      [this, &pre_tokenized_texts, add_special_tokens](std::size_t index) {
        return encode_pre_tokenized_words(
            pre_tokenized_texts[index],
            add_special_tokens,
            true,
            false);
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_char_offsets(
    const std::vector<std::string> & texts,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      texts.size(),
      total_text_bytes(texts),
      [this, &texts, add_special_tokens](std::size_t index) {
        auto encoding = encode_text(texts[index], add_special_tokens, true, false);
        convert_offsets_to_char_offsets(encoding, texts[index]);
        return encoding;
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_char_offsets(
    const std::vector<std::vector<std::string>> & pre_tokenized_texts,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pre_tokenized_texts.size(),
      total_text_bytes(pre_tokenized_texts),
      [this, &pre_tokenized_texts, add_special_tokens](std::size_t index) {
        auto encoding = encode_pre_tokenized_words(
            pre_tokenized_texts[index],
            add_special_tokens,
            true,
            false);
        convert_pre_tokenized_offsets_to_char_offsets(
            encoding,
            pre_tokenized_texts[index]);
        return encoding;
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_pairs(
    const std::vector<std::pair<std::string, std::string>> & pairs,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pairs.size(),
      total_pair_text_bytes(pairs),
      [this, &pairs, add_special_tokens](std::size_t index) {
        const auto & pair = pairs[index];
        return encode_pair_text(
            pair.first,
            pair.second,
            add_special_tokens,
            false);
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_pairs(
    const std::vector<
        std::pair<std::vector<std::string>, std::vector<std::string>>> & pairs,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pairs.size(),
      total_pair_text_bytes(pairs),
      [this, &pairs, add_special_tokens](std::size_t index) {
        const auto & pair = pairs[index];
        return encode_pair_pre_tokenized_words(
            pair.first,
            pair.second,
            add_special_tokens,
            false);
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_pairs_char_offsets(
    const std::vector<std::pair<std::string, std::string>> & pairs,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pairs.size(),
      total_pair_text_bytes(pairs),
      [this, &pairs, add_special_tokens](std::size_t index) {
        const auto & pair = pairs[index];
        auto encoding = encode_pair_text(
            pair.first,
            pair.second,
            add_special_tokens,
            false);
        convert_pair_offsets_to_char_offsets(encoding, pair.first, pair.second);
        return encoding;
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::vector<Encoding> Tokenizer::encode_batch_pairs_char_offsets(
    const std::vector<
        std::pair<std::vector<std::string>, std::vector<std::string>>> & pairs,
    bool add_special_tokens) const {
  auto encodings = map_batch_ordered<Encoding>(
      pairs.size(),
      total_pair_text_bytes(pairs),
      [this, &pairs, add_special_tokens](std::size_t index) {
        const auto & pair = pairs[index];
        auto encoding = encode_pair_pre_tokenized_words(
            pair.first,
            pair.second,
            add_special_tokens,
            false);
        convert_pre_tokenized_pair_offsets_to_char_offsets(
            encoding,
            pair.first,
            pair.second);
        return encoding;
      });
  apply_padding_to_batch(encodings, impl_->padding_);
  return encodings;
}

std::string Tokenizer::decode(
    const std::vector<std::uint32_t> & ids,
    bool skip_special_tokens) const {
  auto decoded_tokens = [this, &ids, skip_special_tokens] {
    std::vector<std::string> tokens;
    for (const auto id : ids) {
      if (id >= impl_->id_to_token_.size()) {
        continue;
      }
      if (skip_special_tokens && id < impl_->special_id_.size() && impl_->special_id_[id]) {
        continue;
      }
      if (!impl_->id_to_token_[id].empty()) {
        tokens.push_back(impl_->id_to_token_[id]);
      }
    }
    return tokens;
  };

  if (!impl_->decoder_steps_.empty()) {
    return join_decoded_tokens(apply_decoder_steps(decoded_tokens(), impl_->decoder_steps_));
  }

  if (impl_->byte_level_decoder_.enabled) {
    return decode_byte_level_tokens(decoded_tokens());
  }

  if (impl_->metaspace_decoder_.enabled) {
    return decode_metaspace_tokens(decoded_tokens(), impl_->metaspace_decoder_);
  }

  if (impl_->wordpiece_decoder_.enabled) {
    return decode_wordpiece_tokens(decoded_tokens(), impl_->wordpiece_decoder_);
  }

  if (impl_->model_type_ == "WordPiece") {
    return join_decoded_tokens_with_spaces(decoded_tokens());
  }

  std::string output;
  for (const auto id : ids) {
    if (id >= impl_->id_to_token_.size()) {
      continue;
    }
    if (skip_special_tokens && id < impl_->special_id_.size() && impl_->special_id_[id]) {
      continue;
    }

    const auto & token = impl_->id_to_token_[id];
    if (token.empty()) {
      continue;
    }
    if (starts_with(token, "##")) {
      output.append(token.substr(2));
    } else if (starts_with(token, "Ġ")) {
      output.push_back(' ');
      output.append(token.substr(2));
    } else {
      if (!output.empty()) {
        output.push_back(' ');
      }
      output.append(token);
    }
  }
  return output;
}

std::vector<std::string> Tokenizer::decode_batch(
    const std::vector<std::vector<std::uint32_t>> & sequences,
    bool skip_special_tokens) const {
  return map_batch_ordered<std::string>(
      sequences.size(),
      total_id_count(sequences),
      [this, &sequences, skip_special_tokens](std::size_t index) {
        return decode(sequences[index], skip_special_tokens);
      });
}

DecodeStream Tokenizer::decode_stream(bool skip_special_tokens) const {
  return DecodeStream(*this, skip_special_tokens);
}

std::optional<std::uint32_t> Tokenizer::token_to_id(const std::string & token) const {
  const auto found = impl_->token_to_id_.find(token);
  if (found != impl_->token_to_id_.end()) {
    return found->second;
  }
  for (const auto & added_token : impl_->added_tokens_) {
    if (added_token.content == token ||
        (!added_token.normalized_content.empty() &&
         added_token.normalized_content == token)) {
      return added_token.id;
    }
  }
  return std::nullopt;
}

std::optional<std::string> Tokenizer::id_to_token(std::uint32_t id) const {
  if (id >= impl_->id_to_token_.size() || impl_->id_to_token_[id].empty()) {
    return std::nullopt;
  }
  return impl_->id_to_token_[id];
}

std::size_t Tokenizer::get_vocab_size() const {
  return impl_->id_to_token_.size();
}

}  // namespace tokenizers_cpp
