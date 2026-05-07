# Model JSON Load Slice

## Source

Upstream target:
`third_party/tokenizers/tokenizers/tests/serialization.rs::{bpe_serde,wordpiece_serde,wordlevel_serde,test_deserialize_long_file}`
at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`.

The Rust tests prove serde roundtrip for standalone model objects and load
success for a large tokenizer JSON file. `tokenizers.cpp` has no public
serialization API yet, so this slice translates those tests into C++ JSON load
acceptance and load-integrity checks only.

## Dependency Decision For This Slice

The JSON parser for this work must come from a local single-header nlohmann JSON
copy in `projects/tokenizers.cpp/include/nlohmann/json.hpp`. Use the existing
header from `projects/magic-tts-ggml-cpp/vendor/nlohmann/json.hpp` as the source
for that copy. Do not use `FetchContent`, a configure-time network download, or
a required system `nlohmann_json` package for default CMake configuration.

This dependency decision does not allow edits to
`projects/tokenizers.cpp/third_party/tokenizers`; that clone remains read-only
reference material.

## Acceptance

C++ tests should use local deterministic JSON fixtures derived from the upstream
serialized forms and load them through the production tokenizer/model JSON path.
They should assert structured load success and selected vocabulary integrity,
not byte-identical JSON reserialization and not encode/decode behavior.

- `serialization.rs::bpe_serde`: accept a real `BPE` model JSON generated from
  `get_byte_level_bpe()`, including upstream field names for vocabulary, merges,
  dropout, unknown-token settings, prefixes/suffixes, byte fallback, and related
  BPE config. Assert the model dispatches as `BPE`, representative
  `token_to_id`/`id_to_token` lookups survive load, and malformed vocab or
  merges fail during load.
- `serialization.rs::wordpiece_serde`: accept a real `WordPiece` model JSON
  generated from `get_bert_wordpiece()`, including `unk_token`,
  `continuing_subword_prefix`, `max_input_chars_per_word`, and `vocab`. Assert
  `WordPiece` dispatch and representative vocabulary lookup integrity.
- `serialization.rs::wordlevel_serde`: accept a real `WordLevel` model JSON
  generated from `WordLevel::from_file("data/gpt2-vocab.json", "<unk>")`,
  including `vocab` and `unk_token`. Assert `WordLevel` dispatch and
  representative vocabulary lookup integrity.
- `serialization.rs::test_deserialize_long_file`: accept the large local
  tokenizer JSON corresponding to
  `data/albert-base-v1-tokenizer.json`. This is a full-graph load smoke for
  top-level tokenizer fields, added-token records, wrapper dispatch, and model
  config retention. Assert load success and a few stable ids/tokens from the
  file; do not call encode/decode as proof of behavior.

Optional slots may be absent or `null` as defined in ADR-0002. Non-null model
objects must reject missing `type`, unsupported `type`, wrong-slot types,
malformed required fields, duplicate or invalid ids, and invalid merge entries
with diagnostics that name the model family or offending field well enough for
fixture triage.

## Assertions To Avoid

- Do not assert C++ can serialize models or tokenizers back to upstream JSON.
- Do not assert BPE merge application, WordPiece greedy matching, WordLevel
  unknown-token fallback during encode, added-token matching,
  normalization, pre-tokenization, post-processing, decoding, truncation,
  padding, offsets, word ids, or masks from these load tests.
- Do not add public component introspection only to satisfy this slice. Prefer
  load success/failure and existing stable vocabulary lookup APIs.

## Remaining Behavior Blockers

- BPE tokenization and decode parity, including merge ranking, byte-level
  behavior, dropout-field load versus stochastic behavior, byte fallback, and
  unknown-token handling.
- WordPiece tokenization parity, including greedy longest-match behavior,
  prefix handling, maximum input characters, and decoder behavior.
- WordLevel encode/decode parity beyond the currently covered unknown fallback
  smoke path.
- Unigram runtime behavior is covered separately in
  `unigram-runtime-boundary.md`; SentencePiece-style tokenizer component
  combinations remain follow-ups.
- Large tokenizer JSON runtime behavior for the components it contains,
  especially Unicode normalization sequences, precompiled maps, added-token
  matching, post-processing, decoder behavior, truncation, padding, offsets, and
  overflow ordering.

`added_tokens.rs::*` remains the next behavior-oriented slice after this model
JSON load coverage lands.
