#include "tokenizers_cpp/unicode_backend.hpp"

#include <cassert>
#include <string>

int main() {
  namespace unicode = tokenizers_cpp::unicode;

  assert(std::string(unicode::backend_name()) == "icu4c");
  assert(unicode::is_letter('A'));
  assert(unicode::is_letter(0x6771U));
  assert(unicode::is_number('9'));
  assert(unicode::is_number(0xFF11U));
  assert(unicode::is_whitespace(' '));
  assert(unicode::is_whitespace(0x3000U));
  assert(unicode::is_control(0x0000U));
  assert(unicode::is_mark(0x0301U));
  assert(unicode::is_punctuation('!'));
  assert(unicode::is_punctuation(0x00BFU));

  assert(unicode::lowercase_utf8("ABC\xCE\x94") == "abc\xCE\xB4");
  assert(unicode::lowercase_utf8("\xCE\x9F\xCE\xA3") == "\xCE\xBF\xCF\x82");
  const auto lowered_sigma = unicode::lowercase_utf8_with_spans(
      "\xCE\x9F\xCE\xA3");
  assert(lowered_sigma.text == "\xCE\xBF\xCF\x82");
  assert(lowered_sigma.byte_spans.size() == lowered_sigma.text.size());
  assert(lowered_sigma.byte_spans[0].start == 0);
  assert(lowered_sigma.byte_spans[0].end == 2);
  assert(lowered_sigma.byte_spans[2].start == 2);
  assert(lowered_sigma.byte_spans[2].end == 4);

  const auto nfd =
      unicode::normalize_utf8("caf\xC3\xA9", unicode::NormalizationForm::Nfd);
  assert(nfd.text == "cafe\xCC\x81");
  assert(nfd.byte_spans.size() == nfd.text.size());
  assert(nfd.byte_spans.front().start == 0);
  assert(unicode::is_punctuation(0x10100U));
  assert(nfd.byte_spans.front().end == std::string("caf\xC3\xA9").size());

  const auto identity =
      unicode::normalize_utf8("plain", unicode::NormalizationForm::Nfc);
  assert(identity.text == "plain");
  assert(identity.byte_spans.size() == identity.text.size());

  return 0;
}
