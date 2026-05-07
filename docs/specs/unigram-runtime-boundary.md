# Unigram Runtime Boundary

## Source

Upstream targets at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `third_party/tokenizers/tokenizers/tests/unigram.rs::test_unigram_from_file`
- `third_party/tokenizers/tokenizers/src/models/unigram/model.rs::test_encode`
- `third_party/tokenizers/tokenizers/src/models/unigram/model.rs::test_unigram_bytefallback`

This is native C++ inference runtime only. It must not add Rust FFI, runtime
shell-out, trainers, training fixtures, stochastic sampling, HTTP/from-pretrained
loading, wrapper APIs, or edits under `third_party/tokenizers`.

## Accepted Behavior

`Tokenizer::from_file` loads these Unigram model fields from tokenizer JSON:

- `vocab`: ordered `[token, score]` entries
- `unk_id`: unsigned id or `null`
- `byte_fallback`: boolean, default `false`
- `fuse_unk`: boolean, default `true` as an internal compatibility extension

The runtime implements the deterministic upstream optimized best-path path:

- Scores are accumulated over byte spans in the original UTF-8 input.
- All vocabulary tokens matching a common prefix at a UTF-8 scalar boundary are
  candidates for the dynamic-programming best path.
- If no single-scalar token exists at a position, `unk_id` is used with
  `min_score - 10.0`.
- Consecutive unknown spans fuse into one raw-token span when `fuse_unk=true`.
- `byte_fallback=true` emits `<0xNN>` tokens for every byte in an unknown span
  only when every byte token exists in vocab; otherwise the raw unknown span is
  emitted with `unk_id`.
- Token offsets remain byte offsets into the original input.

Sampling, `nbest_size`, `alpha`, training, and cache behavior are outside this
boundary.

## Current C++ Coverage

`tests/parity/unigram_test.cpp` covers:

- the real upstream `data/unigram.json` fixture and the exact Japanese token
  sequence from `unigram.rs::test_unigram_from_file`
- best-path preference and fused unknown output using the upstream model.rs
  doc fixture: `abcdacdxx -> abcd, a, cd, xx`
- byte fallback for `é -> <0xC3>, <0xA9>` with the upstream same-span offset
  behavior for each byte fallback token

The tests exercise production `Tokenizer::from_file` and `encode` with local
JSON fixtures. Rust is used only as the upstream reference suite.

## Known Gaps

- Unigram `sample`, `sample_nbest`, `alpha`, and `nbest_size` remain excluded
  because this project targets deterministic inference.
- The raw standalone `Unigram::load(path)` API is not public in C++; tests wrap
  the same model JSON under the tokenizer JSON `model` slot.
- Pre-tokenizer and normalizer combinations around SentencePiece-style Unigram
  tokenizer JSON are future smoke tests after Metaspace and related components
  are ported.
