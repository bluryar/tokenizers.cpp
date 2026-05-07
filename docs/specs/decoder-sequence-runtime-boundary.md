# Decoder Sequence Runtime Boundary

## Source

Semantic references:

- `third_party/tokenizers/tokenizers/src/decoders/mod.rs`
- `third_party/tokenizers/tokenizers/src/decoders/sequence.rs`
- `third_party/tokenizers/tokenizers/src/decoders/byte_fallback.rs`
- `third_party/tokenizers/tokenizers/src/decoders/fuse.rs`
- `third_party/tokenizers/tokenizers/src/decoders/strip.rs`
- `third_party/tokenizers/tokenizers/src/decoders/bpe.rs`
- `third_party/tokenizers/tokenizers/src/decoders/ctc.rs`
- `third_party/tokenizers/tokenizers/src/normalizers/replace.rs`

This slice remains native C++ inference only. It must not add Rust FFI, runtime
shell-out, trainers, HTTP/from-pretrained loading, or edits under
`third_party/tokenizers`.

## Covered Behavior

The tokenizer now parses decoder JSON into an ordered native decoder-step chain.
Direct decoder wrappers and nested `Sequence` wrappers share the same runtime
path. Each step applies Hugging Face's `decode_chain(Vec<String>)` semantics;
the final `Tokenizer::decode` joins the resulting tokens without separators.

Covered runtime steps:

- `ByteLevel`: maps byte-level unicode characters back to bytes and fuses into
  one string.
- `Metaspace`: maps replacement markers back to spaces while honoring
  `prepend_scheme`.
- `WordPiece`: handles continuation prefix removal and optional cleanup.
- `ByteFallback`: converts consecutive `<0xNN>` tokens to UTF-8 or `U+FFFD`
  per invalid byte.
- `Fuse`: joins the current token list into one token.
- `Strip`: strips a single configured UTF-8 scalar from token starts/stops.
- `Replace`: supports `String` and ICU-backed `Regex` patterns per token.
- `BPEDecoder`: replaces the configured word-end suffix with spaces between
  words and nothing on the final token.
- `CTC`: removes adjacent duplicates, drops the pad token, applies the word
  delimiter replacement, and reuses WordPiece cleanup.

## Current C++ Coverage

`tests/parity/decoder_sequence_test.cpp` covers:

- `Sequence(ByteFallback, Fuse)` for valid and invalid UTF-8 byte fallback
  runs, plus stream hardening coverage for valid runs, orphan-byte flushing,
  and no-finalize reset by discarding the stream.
- `Sequence(CTC, Metaspace)` matching upstream `sequence_basic` shape.
- `Sequence(Metaspace, Fuse)` and `Sequence(ByteLevel, Fuse)` stream hardening
  fixtures.
- `Sequence(CTC, Fuse)` for default cleanup of punctuation/contractions and
  custom pad/delimiter configuration with `cleanup=false`.
- `Sequence(Strip, Replace, Fuse)` for per-token strip and string replacement.
- `Sequence(Replace(Regex), Fuse)` through the ICU regex backend.
- Direct `BPEDecoder` runtime behavior.

`tests/parity/json_wrapper_dispatch_test.cpp` continues to cover wrapper
acceptance for decoder `Sequence` JSON.

## Known Gaps

- Broad decoder composition is now structurally supported, but coverage is
  focused on deterministic inference fixtures. Additional upstream CTC and
  decoder edge-case fixtures should be added as they become relevant.
- `Replace` decoder uses ICU regex matching. It does not yet implement
  Rust-regex-specific edge cases beyond the covered ICU-compatible patterns.
- The stream decoder intentionally exposes `step(id)` and `has_pending()` only.
  It has no explicit flush/finalize API for invalid trailing UTF-8; callers
  discard the stream to reset pending state.
