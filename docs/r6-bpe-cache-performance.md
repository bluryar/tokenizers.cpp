# R6 BPE Cache Performance

Date: 2026-05-07

## Status

R6 BPE cache performance hardening is complete.

## Decision

The BPE encode path now uses a private thread-local cache keyed by a
per-tokenizer BPE cache id and normalized input piece text.

The cache stores merge results as BPE symbols:

- token id
- normalized byte start
- normalized byte end

It does not store final `Encoding` offsets. Offsets are still projected through
the current input piece every time, so repeated normalized pieces at different
original byte positions keep correct offsets.

## Boundary

- Cache capacity follows the Rust upstream default shape: 10,000 entries per
  tokenizer cache id per thread.
- Pieces of 256 bytes or longer are not cached.
- Stochastic dropout paths are not cached.
- Runtime `add_tokens` bumps the private BPE cache id because it can extend the
  C++ vocab lookup surface used by BPE tokenization.
- The public tokenizer-centered API does not change.

## Rust Reference

Rust upstream keeps BPE cache state private to the model and currently uses
thread-local cache maps keyed by a per-BPE generation id. This C++ cache follows
the same performance idea while keeping this project's lower-level BPE cache API
private.

## Validation

- `tokenizers_cpp_bpe_config_test` covers no-cross-instance behavior and
  repeated-piece offset preservation.
- Full ICU CTest and no-HF open-source smoke pass with the cache enabled.
