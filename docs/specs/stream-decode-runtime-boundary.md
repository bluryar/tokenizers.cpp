# Stream Decode Runtime Boundary

## Source

Semantic references:

- `third_party/tokenizers/tokenizers/tests/stream.rs`
- `third_party/tokenizers/tokenizers/tests/documentation.rs::streaming_tokenizer`
- `hf-internal-testing/tokenizers-test-data/llama-3-tokenizer.json`
- `hf-internal-testing/tokenizers-test-data/roberta.json`
- `hf-internal-testing/tokenizers-test-data/albert-base-v1-tokenizer.json`

This slice remains native C++ inference only. It must not add Rust FFI, runtime
shell-out, trainers, HTTP/from-pretrained loading, or edits under
`third_party/tokenizers`.

## Covered Behavior

`Tokenizer::decode_stream(skip_special_tokens)` is now available as a small
stateful decode helper. Each `step(id)` decodes one token id through the
tokenizer's normal decoder path, while carrying the limited state needed by the
covered streaming decoders: incomplete UTF-8 bytes, Metaspace first-token
position, and ByteFallback `<0xNN>` byte runs.

The public C++ stream surface intentionally remains step-oriented, matching the
upstream Rust API shape. There is no `flush()` or `finalize()` method. C++ adds
`DecodeStream::has_pending()` so callers can detect that the stream currently
holds incomplete or invalid bytes; callers should discard the stream to reset
that state at an end-of-stream boundary.

The current C++ parity test covers:

- Llama 3 ByteLevel decoder load from the real local tokenizer JSON.
- `stream.rs::test_decode_stream_step_no_panic` for ASCII tokens:
  `A B C D E F G H I J`.
- The same upstream streaming test for Korean byte-fragment tokens, where
  partial token ids return `nullopt` until the buffered bytes form a complete
  UTF-8 scalar.
- `documentation.rs::streaming_tokenizer` for RoBERTa ByteLevel stream
  segments: `This`, ` is`, ` an`, ` example`.
- `documentation.rs::streaming_tokenizer` for ALBERT Metaspace stream
  segments, including the upstream distinction between starting a stream at
  `▁is` (`is`) and seeing it after `▁this` (` is`).
- The `documentation.rs::streaming_tokenizer` ByteFallback partial-byte
  example, where `<0xC3>` returns `nullopt`, `<0xA9>` completes `é`, and an
  orphan continuation byte returns `nullopt` until a later non-byte token
  flushes it as `U+FFFD`.
- R4 hardening coverage for small mixed decoder chains:
  `Sequence(ByteFallback, Fuse)`, `Sequence(Metaspace, Fuse)`, and
  `Sequence(ByteLevel, Fuse)`.
- The no-finalize boundary for pending ByteFallback bytes: `has_pending()` stays
  true after an incomplete trailing byte, and creating a new stream is the
  explicit reset path.
- `stream.rs::test_decoding_with_added_bpe` decode behavior using upstream
  hard-coded ids from the Rust reference.
- The encode-side mutation path from
  `stream.rs::test_decoding_with_added_bpe`: runtime ByteLevel normalizer
  replacement, runtime Split pre-tokenizer replacement, runtime
  `add_tokens`, and subsequent Llama BPE encode/decode parity for the two
  upstream added-token cases.

## Current C++ Coverage

`tests/parity/stream_decode_test.cpp` loads the real local
`llama-3-tokenizer.json` and checks:

- `decode({19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128256}, false)`
  returns `Hey! how is this token: 嗎`.
- `decode({19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128257}, false)`
  returns `Hey! how is this token: д`.
- `with_byte_level_normalizer()`,
  `with_split_pre_tokenizer(<upstream stream.rs regex>)`, and runtime
  `add_tokens({AddedToken{"嗎", normalized=false}})` produce the upstream
  ids, tokens, offsets, word ids, type ids, masks, and decoded text for
  `encode("Hey! how is this token: 嗎", false)`.
- Adding `AddedToken{"д", normalized=true}` after the ByteLevel normalizer
  stores the normalized added-token decode form `Ð´` and produces the upstream
  ids, tokens, offsets, word ids, type ids, masks, and decoded text for
  `encode("Hey! how is this token: д", false)`.
- `decode_stream(false).step(...)` emits exactly the upstream ASCII stream
  segments.
- Korean split-byte tokens return no output for incomplete byte sequences and
  emit `삥`, `뽕`, and `빵` once complete.
- `DecodeStream::has_pending()` is `true` while Korean ByteLevel fragments or
  ByteFallback `<0xNN>` byte runs are buffered, and becomes `false` once a valid
  UTF-8 chunk is emitted or an invalid ByteFallback run is flushed by a later
  non-byte token.
- The real local `roberta.json` emits the upstream documentation stream
  segments for ids `{713, 16, 41, 1246}`.
- The real local `albert-base-v1-tokenizer.json` encodes
  `This is an example` to `{48, 25, 40, 823}` and emits the upstream
  Metaspace stream segments.
- A small local ByteFallback tokenizer fixture pins the upstream documentation
  `None` behavior for partial UTF-8 bytes and the C++ stream boundary for
  orphan-byte replacement before the next regular token.
- Local mixed decoder fixtures pin stream behavior for:
  - `Sequence(ByteFallback, Fuse)`: valid UTF-8 byte runs, orphan continuation
    byte replacement before a later regular token, and no-finalize reset by
    discarding the stream.
  - `Sequence(Metaspace, Fuse)`: first-token and later-token space handling.
  - `Sequence(ByteLevel, Fuse)`: regular word pieces and split UTF-8 bytes.

## Known Gaps

- Runtime mutation coverage is intentionally narrow. The current public
  mutators cover only the ByteLevel normalizer and Split pre-tokenizer shape
  needed by `stream.rs::test_decoding_with_added_bpe`; arbitrary normalizer and
  pre-tokenizer replacement remains a follow-up.
- Normalized added-token matching is accepted for the current ByteLevel
  normalizer fixture. Broader matching after lowercase, Unicode normalization,
  and future ICU-backed normalization remains a follow-up.
- There is intentionally no stream `flush()`/`finalize()` method. Upstream Rust
  exposes only `step(id)`; this C++ port exposes `has_pending()` as the
  non-throwing status channel for incomplete or invalid pending UTF-8.
- Decoder composition now has a shared native `Sequence` path for the covered
  steps in `docs/specs/decoder-sequence-runtime-boundary.md`. Streaming now
  covers direct ByteLevel, Metaspace, and ByteFallback behavior plus the small
  R4 hardening matrix for ByteFallback/Fuse, Metaspace/Fuse, and
  ByteLevel/Fuse. Broader arbitrary mixed decoder chains remain
  downstream-driven follow-up scope.
