# R6 Token-To-Id Map Performance

Date: 2026-05-07

## Status

R6 token-to-id map performance hardening is complete.

## Decision

`Tokenizer::Impl` now keeps a persistent `token_to_id_` map alongside
`id_to_token_`. Loading vocabularies, post-processor special tokens, padding
tokens, and runtime added tokens update both structures together.

The public API does not change. The map is a private implementation cache used
by encode paths, BPE merge loading, and `Tokenizer::token_to_id`.

## Motivation

Before this change, encode and lookup paths rebuilt a full
`std::unordered_map<std::string, uint32_t>` from `id_to_token_` on demand. That
was semantically simple but expensive for realistic 30k-100k token vocabularies,
especially in batch encode and pair encode where the same tokenizer state is
reused repeatedly.

Rust upstream keeps model vocabulary maps as long-lived model state. This change
aligns the C++ runtime shape with that design without exposing lower-level model
APIs.

## Invariants

- `id_to_token_` remains the source of token text by id.
- `token_to_id_` is updated whenever project-owned code writes vocabulary
  entries.
- Replacing a token at an id removes the stale reverse entry if it points to the
  same id.
- Added-token matching and decode semantics are unchanged.
- No public header, dependency, dynamic-library, or CMake network behavior
  changes.

## Validation

- `cmake --build projects/tokenizers.cpp/build-icu`
- `ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure`
- `cmake --build projects/tokenizers.cpp/build-open-source-smoke`
- `ctest --test-dir projects/tokenizers.cpp/build-open-source-smoke --output-on-failure`
- `uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check`
