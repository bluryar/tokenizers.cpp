# R6 Unigram Trie And Cache Performance

Date: 2026-05-07

## Status

R6 Unigram trie/cache performance hardening is complete.

## Decision

The Unigram encode path now builds a private byte trie when tokenizer JSON loads
the Unigram vocabulary. Best-path dynamic programming walks only matching trie
prefixes at each input byte position instead of scanning the full vocabulary.

Repeated normalized pieces also use a private thread-local cache keyed by a
per-tokenizer Unigram cache id and normalized piece text.

The public tokenizer-centered API does not change.

## Boundary

- The trie is a private implementation detail inside `Tokenizer::Impl`.
- Cache entries store token ids, token text, and piece-relative byte offsets.
  Original-input offsets are still projected through the current normalized
  piece during encode.
- Cache capacity is 10,000 entries per tokenizer cache id per thread.
- Pieces of 256 bytes or longer are not cached.
- Runtime `add_tokens` bumps the private Unigram cache id because added tokens
  can extend the reverse vocab lookup used after best-path segmentation.
- The cache does not implement Unigram sampling. Sampling remains outside the
  accepted inference-only scope.

## Rust Reference

Rust upstream uses trie/cache-oriented model state for Unigram inference instead
of repeatedly scanning the complete vocabulary. This C++ slice follows the same
performance shape while keeping lower-level model/cache APIs private.

## Validation

- `tokenizers_cpp_unigram_test` covers the original Unigram fixture, best-path
  segmentation, byte fallback, repeated-piece offset preservation, and cache
  separation across tokenizer instances.
- Full ICU CTest and no-HF open-source smoke pass with the trie/cache path
  enabled.
