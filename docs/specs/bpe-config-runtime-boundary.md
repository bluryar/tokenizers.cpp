# BPE Config Runtime Boundary

## Source

Upstream targets at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_unk_not_fused`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_unk_get_fused`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_with_continuing_subword_prefix`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_byte_fallback`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_byte_fallback_newline`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_ignore_merges`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_cache_is_per_bpe_instance`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_from_file`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_from_file_bad_merges`
- `third_party/tokenizers/tokenizers/src/models/bpe/model.rs::test_bpe_from_file_merge_token_oov`

This is native C++ inference runtime only. It must not add Rust FFI, runtime
shell-out, trainers, training fixtures, HTTP/from-pretrained loading, wrapper
APIs, or edits under `third_party/tokenizers`.

## Accepted Behavior

`Tokenizer::from_file` loads these BPE model fields from tokenizer JSON:

- `unk_token`: string or `null`
- `continuing_subword_prefix`: string or `null`
- `end_of_word_suffix`: string or `null`
- `dropout`: number between `0` and `1` or `null`
- `fuse_unk`: boolean
- `byte_fallback`: boolean
- `ignore_merges`: boolean

`Tokenizer::from_bpe_files(vocab_path, merges_path, options)` is the accepted
C++ public surface for the inference subset of upstream `BPE::from_file`. It
loads `vocab.json` plus `merges.txt` directly and returns a raw BPE tokenizer
with no normalizer, pre-tokenizer, post-processor, decoder, truncation, padding,
or added-token state. `BpeOptions` exposes the same inference config fields
listed above.

`ADR-0004` keeps the BPE public surface tokenizer-centered. This spec therefore
tests `Tokenizer::from_file` and `Tokenizer::from_bpe_files`; it does not require
a public `models::BPE`, `BpeBuilder`, cache resize/clear controls, or exact
stochastic dropout output parity.

The runtime applies the upstream `merge_word` ordering:

- Split the pre-tokenized BPE segment into UTF-8 scalar strings.
- Add `continuing_subword_prefix` to non-first scalars before vocab lookup.
- Add `end_of_word_suffix` to the last scalar before vocab lookup.
- Resolve direct vocab hits before fallback handling.
- If `byte_fallback=true`, emit `<0xNN>` byte tokens only when every byte in
  the candidate exists in vocab.
- If byte fallback fails and `unk_token` is configured, emit the unknown token.
  Consecutive unknowns fuse only when `fuse_unk=true`.
- If `ignore_merges=true` and the full input segment exists in vocab, return
  that token directly; otherwise run normal merge ranking.
- Merge output derivation strips `continuing_subword_prefix` from the right
  merge token when computing the new token, matching upstream BPE builder
  behavior for pairs such as `("a", "##b") -> "ab"`.
- `dropout=null` and `dropout=0.0` keep normal deterministic merges.
- `dropout=1.0` skips every merge and therefore returns the initial scalar
  symbols.
- `0.0 < dropout < 1.0` uses the same probability meaning as upstream BPE
  dropout. Because this is stochastic, stable tests assert only output shape
  invariants: non-empty output, no more pieces than the initial scalar split,
  valid ids, and full input offset coverage.

The same merge/fallback logic is used for direct raw BPE input when no
ByteLevel pre-tokenizer is configured. Offsets remain byte offsets into the
original input after projection through either the raw normalized span or the
current ByteLevel pre-tokenizer alignment.

The C++ runtime does not currently keep a BPE tokenization cache. The accepted
cache boundary is therefore behavioral: independent `Tokenizer` instances with
different merges must tokenize the same input independently, matching upstream's
per-BPE-instance cache isolation requirement.

Added tokens, training, and tokenizer mutation APIs are outside this boundary.

## Current C++ Coverage

`tests/parity/bpe_config_test.cpp` covers:

- non-fused unknown fallback: `accb -> a, <unk>, <unk>, b`
- fused unknown fallback: `accb -> a, <unk>, b`
- missing configured unknown token failure
- `ignore_merges=true` full-vocab hit before merge ranking
- continuing-subword prefix merge: `a + ##b -> ab`
- end-of-word suffix lookup
- ASCII byte fallback plus fallback-to-unk when a byte token is unavailable
- raw non-ByteLevel newline byte fallback: `"\n" -> <0x0A>`
- independent raw BPE instances with different merges cannot pollute each other
- public `Tokenizer::from_bpe_files` success path with `#version: 0.2`
  `merges.txt`
- public `Tokenizer::from_bpe_files` options path, including `dropout=1.0`
- public `Tokenizer::from_bpe_files` bad merges and merge-token OOV failures
- `dropout=null` and `dropout=0.0` preserve deterministic merge output
- `dropout=1.0` skips all merges and returns initial scalar symbols
- `dropout=0.5` shape invariants without asserting an exact stochastic path
- invalid dropout values outside `[0, 1]` are rejected during JSON load
- merge-token out-of-vocabulary failure during JSON model load

The test uses minimal JSON tokenizers with either a direct `ByteLevel`
pre-tokenizer or no pre-tokenizer. This slice exercises production
`Tokenizer::from_file` and `encode` without depending on Rust at runtime.

## Known Gaps

- Stochastic BPE dropout for `0.0 < dropout < 1.0` is covered by shape
  invariants only; exact token sequences are intentionally not fixture-stable.
- A standalone public `models::BPE` class, fluent `BpeBuilder`, and explicit
  cache resize/clear APIs are intentionally deferred by `ADR-0004` unless a
  concrete downstream integration needs them.
- Full Unicode regex and normalization behavior remains owned by the
  pre-tokenizer/normalizer milestones.
