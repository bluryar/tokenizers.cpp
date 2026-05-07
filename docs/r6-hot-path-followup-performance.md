# R6 Hot-Path Follow-Up Performance

Date: 2026-05-07

## Status

Follow-up low-hanging hot-path optimizations are implemented without changing
the tokenizer-centered public API or adding dependencies.

## Changes

- Batch encode/decode now keeps small workloads serial and only fans out to
  `std::async` when both item count and approximate work units are large enough.
  This avoids paying thread-launch overhead for common small-batch inference.
- ICU regex matching now uses a bounded thread-local compiled-regex cache.
  `Split`, `Replace`, and whitespace-style regex users can reuse the compiled
  `URegularExpression` instead of opening and closing it for every call.
- WordPiece inference now builds private initial/continuation byte tries at
  tokenizer load time. Greedy longest-match lookup no longer allocates a
  candidate string for every suffix probe on the hot path.
- Common model ids are cached privately after tokenizer load and after runtime
  `add_tokens`: BPE/WordPiece/WordLevel unknown ids and BPE/Unigram byte
  fallback ids. Runtime logic still falls back to `token_to_id_` when a cached
  id is absent, preserving mutation behavior.

## Boundaries

- No public API changes.
- No new third-party dependency.
- No trainer, sampling, or lower-level model-builder surface.
- Parallelism remains an implementation detail; there is no public
  `TOKENIZERS_PARALLELISM` equivalent yet.

## Measurement

The R6 benchmark matrix now includes:

- `wordpiece_trie_repeated`
- `wordpiece_small_batch`

The real tokenizer JSON rows continue to exercise BERT WordPiece, ICU regex
paths, and batch/decode behavior when local HF test data exists.
