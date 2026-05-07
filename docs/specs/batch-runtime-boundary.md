# Batch Runtime Boundary

## Scope

This slice ports the tokenizer-level batch execution shape for native C++
inference APIs.

Covered behavior:

- ordered `encode_batch` for string inputs
- ordered `encode_batch` for pre-tokenized inputs
- ordered `encode_batch_pairs` for string pairs
- ordered `encode_batch_pairs` for pre-tokenized pairs
- ordered char-offset batch encode APIs
- ordered `decode_batch`
- empty batch handling
- exception propagation from per-item encode/decode work
- batch padding after all top-level encodings are produced

Excluded behavior:

- a public thread-count configuration API
- Rayon-compatible environment variables

## Implementation Notes

Batch work uses an internal ordered map helper. Each item is encoded or decoded
independently into its original index, then callers run the existing shared
post-batch step, such as `BatchLongest` padding, on the completed vector.

This mirrors upstream's tokenizer-level shape: per-item encode/decode can run in
parallel, while batch padding remains a batch-wide operation after the individual
encodings exist.

The helper intentionally stays private to `tokenizer.cpp`. The public API stays
focused on tokenizer behavior, leaving future thread-count, scheduler, or
embedded-runtime policy changes behind one internal boundary.

## Accepted Fixtures

Current acceptance is covered by:

- `tests/parity/tokenizer_api_smoke.cpp` for ordered `decode_batch`,
  char-offset batch APIs, special token skipping, and empty batch handling
- `tests/parity/padding_test.cpp` for ordered batch encoding and batch padding
  across string, pair, pre-tokenized, and overflowing outputs
