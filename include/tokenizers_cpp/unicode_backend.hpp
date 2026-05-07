#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizers_cpp::unicode {

enum class NormalizationForm {
  Nfc,
  Nfd,
  Nfkc,
  Nfkd,
};

struct ByteSpan {
  std::size_t start = 0;
  std::size_t end = 0;
};

struct NormalizedText {
  std::string text;
  std::vector<ByteSpan> byte_spans;
};

struct RegexMatch {
  std::size_t start = 0;
  std::size_t end = 0;
};

const char * backend_name();

bool is_letter(std::uint32_t codepoint);
bool is_number(std::uint32_t codepoint);
bool is_mark(std::uint32_t codepoint);
bool is_nonspacing_mark(std::uint32_t codepoint);
bool is_whitespace(std::uint32_t codepoint);
bool is_control(std::uint32_t codepoint);
bool is_punctuation(std::uint32_t codepoint);

std::string lowercase_utf8(std::string_view text);
NormalizedText lowercase_utf8_with_spans(std::string_view text);
NormalizedText normalize_utf8(std::string_view text, NormalizationForm form);
std::vector<RegexMatch> find_regex_matches_utf8(
    std::string_view text,
    std::string_view pattern);

}  // namespace tokenizers_cpp::unicode
