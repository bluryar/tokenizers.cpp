# Char Offset Runtime Boundary

## Scope

This slice ports the tokenizer-level char-offset API shape for native C++
inference while keeping byte offsets as the default.

Covered behavior:

- `encode_char_offsets(text)`
- `encode_char_offsets(pre_tokenized_words)`
- `encode_pair_char_offsets(text_a, text_b)`
- `encode_pair_char_offsets(pre_tokenized_a, pre_tokenized_b)`
- `encode_batch_char_offsets(texts)`
- `encode_batch_char_offsets(pre_tokenized_texts)`
- `encode_batch_pairs_char_offsets(pairs)`
- `encode_batch_pairs_char_offsets(pre_tokenized_pairs)`
- recursive conversion of overflowing encodings
- special-token and padding offsets remain `(0, 0)`
- real-tokenizer char-offset coverage over BERT `Sequence(NFD, Lowercase,
  StripAccents)`, Whitespace pre-tokenization, WordPiece, TemplateProcessing,
  truncation, overflowing, and fixed padding
- real-tokenizer batch char-offset coverage over Llama
  `Sequence(Split, ByteLevel(use_regex=false))` with truncation, overflowing,
  template insertion, and fixed padding

Excluded behavior:

- a generic `EncodeInput` sum type; the C++ API keeps explicit overloads
- `OffsetType::None`

## Implementation Notes

The default encode APIs continue to return UTF-8 byte offsets. The char-offset
APIs reuse the same encode, truncation, post-processing, and padding paths, then
project token offsets from UTF-8 byte indexes to scalar-value indexes.

Pair offsets are converted against their own sequence. Pre-tokenized offsets are
converted against the corresponding supplied word, preserving the existing C++
contract that pre-tokenized offsets are word-relative.

Batch char-offset APIs run the conversion before shared `BatchLongest` padding,
so padding still remains a batch-wide operation after individual encodings are
produced.

## Accepted Fixtures

`tests/parity/tokenizer_api_smoke.cpp` verifies:

- default byte offsets for `héllo 世界`
- char offsets for `héllo 世界`
- char offsets for pre-tokenized single and pair inputs
- ordered char-offset batch outputs for string and pre-tokenized inputs
- ordered char-offset pair batch outputs for string and pre-tokenized pairs

`tests/parity/llama_encode_test.cpp` extends this API coverage through the
real local `llama-3-tokenizer.json` path:

- `encode_batch_char_offsets({"café 東京 １２3", "Hello"}, true)` over a
  temporary truncation + fixed-padding JSON copy, checking UTF-8 byte offsets
  projected to scalar indexes across main and overflowing encodings
- `encode_batch_pairs_char_offsets(...)` over the same temporary JSON, checking
  ordered pair-batch outputs after pair truncation, template insertion, and
  fixed padding

`tests/parity/bert_wordpiece_added_tokens_test.cpp` extends this API coverage
through the real local `bert-wiki.json` path:

- `encode_char_offsets("Héllo world test token", true)` over temporary
  truncation + fixed-padding JSON, checking normalized WordPiece char offsets
  and recursive overflow conversion for `token -> tok / ##en`
- pre-tokenized `encode_char_offsets({"Héllo", "world", "test", "token"},
  true)`, preserving word-relative char offsets
- `encode_pair_char_offsets(...)` and `encode_batch_pairs_char_offsets(...)`
  for raw and pre-tokenized pair inputs after pair truncation and
  TemplateProcessing `[CLS]/[SEP]` insertion
