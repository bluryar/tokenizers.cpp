# R4-H2 BPE Public Surface Review

Date: 2026-05-07

## Verdict

R4-H2 BPE public surface review is complete.

The accepted direction is to keep `tokenizers.cpp` tokenizer-centered. Public
BPE access remains `Tokenizer::from_file` for complete tokenizer JSON and
`Tokenizer::from_bpe_files` for local `vocab.json` plus `merges.txt` inference.
Standalone public `models::BPE`, fluent builders, explicit cache controls, and
exact stochastic dropout controls are deferred until a concrete downstream
consumer needs them.

The formal decision is recorded in
`docs/adr/ADR-0004-bpe-public-surface.md`.

## Accepted Public BPE Surface

- `Tokenizer::from_file(path)`
- `Tokenizer::from_bpe_files(vocab_path, merges_path, options={})`
- `BpeOptions` for inference-time BPE fields:
  - `unk_token`
  - `continuing_subword_prefix`
  - `end_of_word_suffix`
  - `dropout`
  - `fuse_unk`
  - `byte_fallback`
  - `ignore_merges`

## Explicitly Deferred

- `tokenizers_cpp::models::BPE`
- `BpeBuilder` or other fluent model builders
- public cache resize, clear, capacity, or inspection APIs
- seeded stochastic dropout controls or exact random-output parity tests
- broad Rust-style component mutation APIs

## Current Coverage

The existing BPE test and spec boundary already covers the public surface needed
for downstream inference:

- tokenizer JSON BPE load
- raw `vocab.json` plus `merges.txt` file load
- `#version: 0.2` merge-file handling
- invalid merge lines and merge-token out-of-vocabulary failures
- unknown fallback with and without `fuse_unk`
- byte fallback and fallback-to-unknown behavior
- continuing-subword prefix and end-of-word suffix handling
- `ignore_merges`
- dropout `null`, `0.0`, `1.0`, invalid values, and stochastic shape
  invariants
- independent tokenizer instance behavior so future cache work cannot leak
  tokenization results across instances

## Re-Open Triggers

Revisit the decision only for a concrete downstream C++/GGML integration that
cannot use the tokenizer-centered API. Good triggers include:

- in-memory vocab/merges construction is required by a model loader
- measured BPE cache behavior needs public tuning
- a consumer needs reproducible seeded dropout for an accepted inference mode
- a downstream project needs typed model inspection that cannot be represented
  through `token_to_id`, `id_to_token`, and tokenizer metadata

## Verification

This review did not change runtime code. The relevant runtime baseline remains:

```sh
cmake --build projects/tokenizers.cpp/build-icu
ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure
uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check
```

Latest local result:

- C++ build passed.
- CTest passed: 26/26.
- Fixture generator check passed.
- `third_party/tokenizers` and `third_party/icu` worktrees were clean.
