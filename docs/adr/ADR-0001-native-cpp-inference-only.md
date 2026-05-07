# ADR-0001: Native C++ Inference-Only Tokenizers Port

## Status

Accepted

## Context

`ggbond` needs tokenizer runtime behavior that can be used by future pure C++
inference engines. Hugging Face `tokenizers` already has the reference behavior
and tests, but this project should not add Rust FFI or training dependencies to
the runtime surface.

## Decision

- Implement `tokenizers.cpp` as a native C++ library.
- Use `third_party/tokenizers/tokenizers` only as a read-only reference.
- Preserve inference behavior for local tokenizer JSON loading, encode,
  encode-pair, decode, added tokens, truncation, padding, and runtime
  components.
- Exclude trainers, training tests, HTTP/from-pretrained loading, benches, and
  Python/Node/WASM wrappers.
- Keep the public C++ API close to Hugging Face runtime names and fields.

## Dependency Policy

Pinned dependency defaults:

- JSON: a local single-header copy of nlohmann JSON `3.12.0` at
  `include/nlohmann/json.hpp`, copied from
  `projects/magic-tts-ggml-cpp/vendor/nlohmann/json.hpp`
- Current Unicode normalization work is implemented inside the native C++
  runtime for the covered inference slices. Do not depend on system dynamic
  ICU libraries.
- Future full Unicode parity may use ICU4C under
  `docs/adr/ADR-0003-vendored-static-icu4c-unicode-backend.md`: project-owned,
  pinned, and capable of a production static build with explicit ICU data
  packaging. ICU may be used as a development-time reference tool.
- Future regex-compatible split/replace work may use RE2, but only as
  project-owned vendored/static source or an explicit optional feature.

JSON parsing must be available from the local tree. CMake configure must not
download nlohmann JSON with `FetchContent`, contact the network, or require a
system package for the default build. The default build must also not succeed
only because it discovers system dynamic Unicode or regex libraries. The local
copy is project-owned third-party source; it is separate from the read-only
upstream `third_party/tokenizers` reference clone.

Only dependencies needed by the current implementation should be linked or
included by targets. The remaining pinned dependencies are activated when the
corresponding normalizer/pre-tokenizer components land.

## Consequences

- Runtime consumers will not need Rust, Python, or network access.
- Parity work must be test-driven because tokenizer behavior has many offset
  and Unicode edge cases.
- The project will carry explicit skip records for upstream trainer and HTTP
  tests instead of pretending they are unimplemented runtime gaps.
