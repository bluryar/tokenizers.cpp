#include "tokenizers_cpp/unicode_backend.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <unicode/casemap.h>
#include <unicode/edits.h>
#include <unicode/uchar.h>
#include <unicode/ucnv.h>
#include <unicode/uregex.h>
#include <unicode/ustring.h>
#include <unicode/unorm2.h>
#include <unicode/utypes.h>

namespace tokenizers_cpp::unicode {
namespace {

void require_success(UErrorCode status, const char * operation) {
  if (U_FAILURE(status)) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + u_errorName(status));
  }
}

std::vector<UChar> utf8_to_uchar(std::string_view text) {
  UErrorCode status = U_ZERO_ERROR;
  int32_t required = 0;
  u_strFromUTF8(
      nullptr,
      0,
      &required,
      text.data(),
      static_cast<int32_t>(text.size()),
      &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    require_success(status, "u_strFromUTF8 sizing");
  }

  status = U_ZERO_ERROR;
  std::vector<UChar> output(static_cast<std::size_t>(required));
  u_strFromUTF8(
      output.data(),
      required,
      nullptr,
      text.data(),
      static_cast<int32_t>(text.size()),
      &status);
  require_success(status, "u_strFromUTF8");
  return output;
}

std::string uchar_to_utf8(const UChar * data, int32_t length) {
  UErrorCode status = U_ZERO_ERROR;
  int32_t required = 0;
  u_strToUTF8(nullptr, 0, &required, data, length, &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    require_success(status, "u_strToUTF8 sizing");
  }

  status = U_ZERO_ERROR;
  std::string output(static_cast<std::size_t>(required), '\0');
  u_strToUTF8(output.data(), required, nullptr, data, length, &status);
  require_success(status, "u_strToUTF8");
  return output;
}

std::uint32_t utf8_codepoint_at(std::string_view text, std::size_t pos) {
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
  if (pos + 3 < text.size()) {
    return ((lead & 0x07U) << 18U) |
        ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 12U) |
        ((static_cast<unsigned char>(text[pos + 2]) & 0x3FU) << 6U) |
        (static_cast<unsigned char>(text[pos + 3]) & 0x3FU);
  }
  return 0xFFFDU;
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
  return 4;
}

std::vector<std::size_t> uchar_to_utf8_byte_offsets(std::string_view text) {
  std::vector<std::size_t> offsets;
  offsets.reserve(text.size() + 1);
  for (std::size_t pos = 0; pos < text.size();) {
    const auto codepoint = utf8_codepoint_at(text, pos);
    const auto utf8_length = std::min(
        utf8_codepoint_length(static_cast<unsigned char>(text[pos])),
        text.size() - pos);
    offsets.push_back(pos);
    if (codepoint > 0xFFFFU) {
      offsets.push_back(pos);
    }
    pos += utf8_length;
  }
  offsets.push_back(text.size());
  return offsets;
}

const UNormalizer2 * normalizer_for(NormalizationForm form) {
  UErrorCode status = U_ZERO_ERROR;
  const UNormalizer2 * normalizer = nullptr;
  switch (form) {
    case NormalizationForm::Nfc:
      normalizer = unorm2_getNFCInstance(&status);
      break;
    case NormalizationForm::Nfd:
      normalizer = unorm2_getNFDInstance(&status);
      break;
    case NormalizationForm::Nfkc:
      normalizer = unorm2_getNFKCInstance(&status);
      break;
    case NormalizationForm::Nfkd:
      normalizer = unorm2_getNFKDInstance(&status);
      break;
  }
  require_success(status, "unorm2_get*Instance");
  return normalizer;
}

}  // namespace

const char * backend_name() {
  return "icu4c";
}

bool is_letter(std::uint32_t codepoint) {
  const auto category = u_charType(static_cast<UChar32>(codepoint));
  return category == U_UPPERCASE_LETTER ||
      category == U_LOWERCASE_LETTER ||
      category == U_TITLECASE_LETTER ||
      category == U_MODIFIER_LETTER ||
      category == U_OTHER_LETTER;
}

bool is_number(std::uint32_t codepoint) {
  const auto category = u_charType(static_cast<UChar32>(codepoint));
  return category == U_DECIMAL_DIGIT_NUMBER ||
      category == U_LETTER_NUMBER ||
      category == U_OTHER_NUMBER;
}

bool is_mark(std::uint32_t codepoint) {
  const auto category = u_charType(static_cast<UChar32>(codepoint));
  return category == U_NON_SPACING_MARK ||
      category == U_ENCLOSING_MARK ||
      category == U_COMBINING_SPACING_MARK;
}

bool is_nonspacing_mark(std::uint32_t codepoint) {
  return u_charType(static_cast<UChar32>(codepoint)) == U_NON_SPACING_MARK;
}

bool is_whitespace(std::uint32_t codepoint) {
  return u_isUWhiteSpace(static_cast<UChar32>(codepoint)) != 0;
}

bool is_control(std::uint32_t codepoint) {
  const auto category = u_charType(static_cast<UChar32>(codepoint));
  return category == U_CONTROL_CHAR ||
      category == U_FORMAT_CHAR ||
      category == U_UNASSIGNED ||
      category == U_PRIVATE_USE_CHAR ||
      category == U_SURROGATE;
}

bool is_punctuation(std::uint32_t codepoint) {
  const auto category = u_charType(static_cast<UChar32>(codepoint));
  return category == U_DASH_PUNCTUATION ||
      category == U_START_PUNCTUATION ||
      category == U_END_PUNCTUATION ||
      category == U_CONNECTOR_PUNCTUATION ||
      category == U_OTHER_PUNCTUATION ||
      category == U_INITIAL_PUNCTUATION ||
      category == U_FINAL_PUNCTUATION;
}

std::string lowercase_utf8(std::string_view text) {
  return lowercase_utf8_with_spans(text).text;
}

NormalizedText lowercase_utf8_with_spans(std::string_view text) {
  UErrorCode status = U_ZERO_ERROR;
  int32_t required = icu::CaseMap::utf8ToLower(
      "",
      0,
      text.data(),
      static_cast<int32_t>(text.size()),
      nullptr,
      0,
      nullptr,
      status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    require_success(status, "CaseMap::utf8ToLower sizing");
  }

  status = U_ZERO_ERROR;
  icu::Edits edits;
  NormalizedText output;
  output.text.resize(static_cast<std::size_t>(required));
  icu::CaseMap::utf8ToLower(
      "",
      0,
      text.data(),
      static_cast<int32_t>(text.size()),
      output.text.data(),
      required,
      &edits,
      status);
  require_success(status, "CaseMap::utf8ToLower");

  output.byte_spans.assign(output.text.size(), ByteSpan{0, text.size()});
  auto iterator = edits.getFineIterator();
  status = U_ZERO_ERROR;
  while (iterator.next(status)) {
    const auto source_start = static_cast<std::size_t>(iterator.sourceIndex());
    const auto source_length = static_cast<std::size_t>(iterator.oldLength());
    const auto destination_start =
        static_cast<std::size_t>(iterator.destinationIndex());
    const auto destination_length =
        static_cast<std::size_t>(iterator.newLength());
    const auto destination_end =
        std::min(destination_start + destination_length, output.byte_spans.size());
    if (iterator.hasChange()) {
      const auto span = ByteSpan{source_start, source_start + source_length};
      for (std::size_t index = destination_start; index < destination_end; ++index) {
        output.byte_spans[index] = span;
      }
    } else {
      for (std::size_t index = destination_start; index < destination_end; ++index) {
        const auto offset = index - destination_start;
        output.byte_spans[index] =
            ByteSpan{source_start + offset, source_start + offset + 1};
      }
    }
  }
  require_success(status, "Edits::Iterator::next");
  return output;
}

NormalizedText normalize_utf8(std::string_view text, NormalizationForm form) {
  const auto input = utf8_to_uchar(text);
  UErrorCode status = U_ZERO_ERROR;
  const auto * normalizer = normalizer_for(form);
  int32_t required = unorm2_normalize(
      normalizer,
      input.data(),
      static_cast<int32_t>(input.size()),
      nullptr,
      0,
      &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    require_success(status, "unorm2_normalize sizing");
  }

  status = U_ZERO_ERROR;
  std::vector<UChar> normalized(static_cast<std::size_t>(required));
  unorm2_normalize(
      normalizer,
      input.data(),
      static_cast<int32_t>(input.size()),
      normalized.data(),
      required,
      &status);
  require_success(status, "unorm2_normalize");

  NormalizedText output;
  output.text = uchar_to_utf8(normalized.data(), required);
  output.byte_spans.reserve(output.text.size());
  for (std::size_t index = 0; index < output.text.size(); ++index) {
    output.byte_spans.push_back(ByteSpan{0, text.size()});
  }
  return output;
}

std::vector<RegexMatch> find_regex_matches_utf8(
    std::string_view text,
    std::string_view pattern) {
  if (text.empty()) {
    return {};
  }

  const auto pattern_uchars = utf8_to_uchar(pattern);
  UErrorCode status = U_ZERO_ERROR;
  UParseError parse_error{};
  URegularExpression * regex = uregex_open(
      pattern_uchars.data(),
      static_cast<int32_t>(pattern_uchars.size()),
      0,
      &parse_error,
      &status);
  require_success(status, "uregex_open");

  const auto input = utf8_to_uchar(text);
  uregex_setText(
      regex,
      input.data(),
      static_cast<int32_t>(input.size()),
      &status);
  if (U_FAILURE(status)) {
    uregex_close(regex);
    require_success(status, "uregex_setText");
  }

  const auto byte_offsets = uchar_to_utf8_byte_offsets(text);
  std::vector<RegexMatch> matches;
  while (uregex_findNext(regex, &status)) {
    const auto start = uregex_start(regex, 0, &status);
    const auto end = uregex_end(regex, 0, &status);
    if (U_FAILURE(status)) {
      uregex_close(regex);
      require_success(status, "uregex_findNext");
    }
    if (start >= 0 && end >= start &&
        static_cast<std::size_t>(end) < byte_offsets.size()) {
      const auto byte_start = byte_offsets[static_cast<std::size_t>(start)];
      const auto byte_end = byte_offsets[static_cast<std::size_t>(end)];
      if (byte_start < byte_end) {
        matches.push_back(RegexMatch{byte_start, byte_end});
      }
    }
  }
  if (U_FAILURE(status)) {
    uregex_close(regex);
    require_success(status, "uregex_findNext");
  }

  uregex_close(regex);
  return matches;
}

}  // namespace tokenizers_cpp::unicode
