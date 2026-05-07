# Unicode Normalizer Runtime Boundary

Status: accepted for the current R2 runtime slice.

## Scope

This slice covers inference-time JSON load and execution for the simple
normalizer wrappers that now route through the project-owned ICU backend:

- `NFC`
- `NFD`
- `NFKC`
- `NFKD`
- `Nmt`
- `Prepend`
- `Strip`
- `StripAccents`
- `Lowercase`

The accepted execution path is native C++ only. Rust is used only as an
upstream reference during development; there is no Rust FFI or runtime shell-out.

## Runtime Rules

- `NFC`, `NFD`, `NFKC`, and `NFKD` call the vendored ICU backend.
- `NFC` and `NFKC` normalize a base codepoint plus following nonspacing marks as
  a chunk, allowing common composed outputs such as `e + U+0301 -> é` while
  projecting all produced bytes back to the merged source byte span.
- `NFD` and `NFKD` keep the existing per-codepoint projection rule, which is the
  stable behavior needed by BERT and ALBERT/SentencePiece fixtures.
- `Nmt` removes the exact upstream ASCII/control ranges and maps the upstream
  NMT whitespace set to a normal ASCII space while preserving each changed
  codepoint's original byte span.
- `Prepend` inserts its configured prefix only when the normalized text is
  non-empty. Inserted prefix bytes are projected to the first normalized
  codepoint's original span, matching upstream's alignment behavior.
- `Strip` removes leading and/or trailing ICU Unicode whitespace and preserves
  byte-offset projection for the remaining text.
- Ordered `Sequence` normalizers preserve the upstream component order.

## C++ Coverage

- `tests/parity/replace_normalizer_test.cpp` covers:
  - `Replace(String)`
  - `Replace(Regex)` through ICU regex
  - `Sequence(Strip, NFC)` with composed Unicode output and original-byte
    offsets
  - direct `NFKC` with compatibility expansion and original-byte offsets
  - direct `Nmt` removal/mapping behavior
  - direct `Prepend` non-empty insertion and empty-input no-op behavior
- `tests/parity/post_processor_sequence_test.cpp` covers the real local
  GPT-style `tokenizer.json` normalizer `Sequence(Strip, NFC)` before
  ByteLevel/BPE/post-processing.
- Existing BERT and ALBERT/SentencePiece parity tests cover `NFD`, `NFKD`,
  `StripAccents`, and `Lowercase` in real tokenizer chains.

## Known Gaps

- Broader normalizer wrappers such as `ByteLevel` normalizer and `Prepend` in
  more complex SentencePiece chains are not included in this slice.
- Cross-base canonical reordering beyond the accepted base-plus-mark chunking is
  deferred until an upstream parity fixture requires byte-exact behavior there.
- Serialization output parity is not required; this project currently loads and
  executes tokenizer JSON for inference.
