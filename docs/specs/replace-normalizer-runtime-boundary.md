# Replace Normalizer Runtime Boundary

## Scope

This slice ports the inference runtime behavior for serialized
`{"type":"Replace"}` normalizers.

Covered behavior:

- Direct `Replace` normalizer JSON parsing.
- `Sequence` normalizers containing `Replace`.
- `pattern: {"String": ...}` literal replacement.
- `pattern: {"Regex": ...}` replacement through the shared Unicode backend.
  The default vendored ICU build executes the regex.
- Replacement text is inserted by the normalizer layer, not by ICU, so original
  byte-span projection stays centralized in `tokenizer.cpp`.
- Unigram/SentencePiece encode-side coverage for ids, tokens, offsets, word
  ids, type ids, special-token masks, and attention masks.
- Default-ICU supplementary-plane regex offset coverage for `\p{P}+`.
- ByteLevel/BPE encode-side composition for the accepted generic ICU regex
  fixture. The BPE path consumes normalizer output through span-preserving
  ByteLevel alignment, so replacement bytes can still project back to original
  input byte spans.

This remains inference-only: no Rust FFI, no runtime shell-out, no training,
no HTTP/from-pretrained loading, and no edits under `third_party/tokenizers`.

## Fixtures

`tests/parity/replace_normalizer_test.cpp` covers:

- `normalizers::replace::tests::test_replace` shape:
  string-pattern replacement, `String("_") -> " "`.
- `normalizers::replace::tests::test_replace_regex` shape:
  regex whitespace collapse, `Regex("\\s+") -> " "`, in the default ICU build.
- A default-ICU supplementary-plane punctuation replacement:
  `Regex("\\p{P}+") -> " "` over `hello𐄀world`.
- `tests/parity/byte_level_bpe_test.cpp` adds the same replacement before
  ByteLevel/GPT-2 BPE, checking Rust-derived ids, tokens, offsets, word ids,
  masks, and ByteLevel decode.

The direct tests use a minimal Unigram tokenizer with `WhitespaceSplit`; the
ByteLevel/BPE fixture verifies the same normalizer machinery through the BPE
runtime path.

## Known Gaps

- `Replace` decoder behavior is not covered yet.
- Full Rust regex syntax parity is not claimed. ICU regex is the accepted
  project-owned default backend, so differences between Rust `regex` and ICU
  regex should be handled fixture by fixture.
- Broader `Replace` composition with ByteLevel/BPE beyond the accepted
  supplementary-plane punctuation fixture remains a follow-up.

## Verification

- `ctest --test-dir build-icu -R tokenizers_cpp_replace_normalizer_test --output-on-failure`
- Upstream reference:
  `cargo test normalizers::replace::tests`
