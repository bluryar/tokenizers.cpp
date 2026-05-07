# ADR-0003: Vendored Static ICU4C Unicode Backend

## Status

Accepted

## Context

Early `tokenizers.cpp` runtime slices use self-contained C++ Unicode tables and
targeted UTF-8 scanning so inference parity can advance without external
runtime dependencies. That approach has worked for accepted fixtures, but full
Hugging Face tokenizer parity eventually needs broad Unicode normalization,
case mapping, category properties, and regex behavior such as `\p{L}`,
`\p{N}`, and lookahead.

The project requirement is not to avoid libraries altogether. The real
deployment risk is binding `tokenizers.cpp` to operating-system ICU packages or
runtime `.so` / `.dll` availability and version drift.

## Decision

`tokenizers.cpp` may add ICU4C as a project-owned dependency, but only under
these rules:

- ICU4C must be vendored or otherwise pinned to an exact project-controlled
  source version.
- The production inference path must support a static build that does not
  require system `libicu*.so`, `.dylib`, or `.dll` files at runtime.
- `find_package(ICU)` or similar discovery of system ICU is not allowed for the
  default build.
- Dynamic ICU linking is allowed only as an explicit development or validation
  mode, not as the default or release portability path.
- ICU data must be explicitly packaged. The project must not accidentally link
  only tiny `stubdata` and assume full Unicode behavior is available.
- The preferred ICU data strategy is static data linked into the final library
  or executable. An embedded `.dat` archive initialized by the runtime may be
  accepted after a focused ADR update. A loose external `.dat` file is a
  deployment fallback, not the default.
- Runtime tokenizer code should call a project-local Unicode backend interface
  rather than scattering direct ICU calls through model, normalizer, and
  pre-tokenizer code.

The initial backend shape should support:

- Unicode normalization: NFC, NFD, NFKC, NFKD.
- Lowercase and casefold behavior needed by tokenizer normalizers.
- General category and binary property checks for whitespace, control,
  punctuation, letter, number, and combining mark behavior.
- Regex split/replace behavior needed by accepted tokenizer JSON components,
  including Unicode properties and lookahead.
- Offset projection from normalized text back to original UTF-8 byte offsets.

## Non-Goals

- ICU does not change ADR-0001's inference-only boundary.
- ICU does not authorize trainers, HTTP/from-pretrained loading, Rust FFI,
  Python/Node/WASM wrappers, or runtime shell-out.
- ICU adoption removes the previous self-contained Unicode backend from the
  build. The ICU-backed runtime is the only supported Unicode path.
- ICU adoption does not require accepting every upstream regex component before
  a C++ parity test and spec boundary exist.

## Build Policy

CMake must use explicit options rather than implicit system probing:

- vendored ICU is the default and supported Unicode backend
- `TOKENIZERS_CPP_ICU_VENDOR=ON`
- `TOKENIZERS_CPP_ICU_SHARED=OFF` for production portability

`TOKENIZERS_CPP_USE_ICU=OFF` is retired from the supported build matrix.

The static ICU path should define the necessary ICU static-link compile
definitions, keep ICU data packaging visible in build logs, and include an
audit test that fails if production binaries depend on system ICU shared
libraries.

## Consequences

- Full Unicode and regex parity can move from fragile hand-written tables to a
  mature backend.
- Build time, source size, and final binary size will increase when ICU is
  enabled.
- Cross-compilation becomes more complex because ICU host tools and target
  libraries/data must be managed deliberately.
- The project can remain portable across operating systems by avoiding runtime
  dependence on OS-provided ICU dynamic libraries.
