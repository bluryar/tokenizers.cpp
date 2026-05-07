# ADR-0004: BPE Public Surface

## Status

Accepted

## Context

The core inference milestone already supports BPE inference through local
tokenizer JSON and through `Tokenizer::from_bpe_files(vocab_path, merges_path,
options)`. The accepted runtime behavior covers merge ranking, unknown fallback,
byte fallback, continuing-subword prefix, end-of-word suffix, `ignore_merges`,
dropout field validation, deterministic dropout boundaries, stochastic dropout
shape invariants, merge-file validation, and no-cross-instance cache behavior.

The upstream Rust crate also exposes lower-level model and builder APIs such as
`models::BPE`, `BpeBuilder`, and cache configuration. Copying that internal API
surface into C++ would increase the amount of public API to stabilize before a
downstream GGML/C++ consumer has shown that it needs those entry points.

## Decision

Keep the public C++ BPE surface tokenizer-centered.

Accepted public entry points:

- `Tokenizer::from_file(path)` for complete tokenizer JSON.
- `Tokenizer::from_bpe_files(vocab_path, merges_path, options={})` for the
  inference subset of upstream `BPE::from_file`.
- `BpeOptions` for runtime inference fields that already affect tokenization:
  `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `dropout`,
  `fuse_unk`, `byte_fallback`, and `ignore_merges`.

Do not expose these as public API for now:

- standalone `tokenizers_cpp::models::BPE`
- fluent `BpeBuilder`
- explicit cache resize, clear, or capacity APIs
- seeded or exact-sequence stochastic dropout controls
- general Rust-style tokenizer component builder and mutation APIs

The implementation may keep private BPE helpers and private model state as
needed. Private helpers are not compatibility promises.

## Promotion Criteria

Re-open this ADR only when a concrete downstream integration needs a lower-level
BPE entry point that cannot be served cleanly by `Tokenizer::from_file` or
`Tokenizer::from_bpe_files`.

Any proposal to expose a lower-level BPE API must include:

- the downstream call site that needs it
- a C++ header shape and ownership model
- tests that exercise the public API rather than private helpers
- compatibility notes for vocab/merges loading and tokenizer JSON loading
- a cache policy, if cache controls are exposed
- a dropout policy that does not make stable tests depend on random exact output

Until those criteria are met, R4 BPE work should add parity fixtures or docs
without growing the public API.

## Verification Boundary

The current public surface is considered covered by:

- JSON BPE model loading through `Tokenizer::from_file`
- raw file loading through `Tokenizer::from_bpe_files`
- valid and invalid merge-file tests
- merge-token out-of-vocabulary rejection
- deterministic BPE config fixtures
- dropout `null`, `0.0`, `1.0`, and stochastic shape-invariant fixtures
- independent tokenizer instance behavior for future cache-safety regression
  checks

Exact stochastic dropout sequences, cache capacity behavior, and Rust builder
ergonomics are not acceptance criteria for tokenizer-centered inference.

## Consequences

- Downstream C++ model runtimes get a small and stable API centered on loading
  real tokenizer assets.
- The project avoids copying Rust internal builder ergonomics before there is a
  consumer-driven need.
- Future BPE cache work can stay an implementation detail unless performance
  profiling proves that public cache controls are necessary.
- If a lower-level API is later needed, it will be added deliberately with a new
  ADR update instead of emerging accidentally through hardening tests.
