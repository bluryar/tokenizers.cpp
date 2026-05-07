# R4-H1 Minimal Hardening Freeze

Date: 2026-05-07

## Verdict

R4-H1 is complete.

The goal of this slice was to pin the smallest high-value hardening edges after
the core inference milestone without expanding the public API. The completed
scope covers stream mixed-decoder behavior, the no-finalize stream boundary,
and broader ALBERT/SentencePiece `Precompiled` charsmap validation.

## Completed Scope

- Stream mixed decoder coverage:
  - `Sequence(ByteFallback, Fuse)`
  - `Sequence(Metaspace, Fuse)`
  - `Sequence(ByteLevel, Fuse)`
- Stream finalization boundary:
  - no `flush()` or `finalize()` public API
  - `DecodeStream::has_pending()` exposes incomplete state
  - callers discard the stream and create a new stream to reset pending bytes
- ALBERT `Precompiled` charsmap breadth:
  - no-break and BOM separator boundaries
  - chained zero-width joiner/non-joiner/space deletion
  - compatibility expansion from a single original byte span
  - expansion followed by joiner splitting with byte-offset projection

## Verification

The accepted R4-H1 baseline is:

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

## Remaining R4 Work

- BPE public model/builder/cache/dropout strategy remains downstream-driven.
- Additional `Precompiled` charsmap validation should be added only when new
  real SentencePiece tokenizer JSON fixtures demand it.
- Rare ICU offset projection cases remain follow-up work for concrete fixtures.
- Lower-level model/component public APIs remain private unless a downstream
  integration needs them.
