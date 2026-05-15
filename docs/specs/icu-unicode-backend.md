# ICU Unicode Backend Infrastructure

## Scope

This slice introduces the project-local Unicode backend boundary accepted in
`docs/adr/ADR-0003-vendored-static-icu4c-unicode-backend.md`.

The default build uses the ICU backend and expects ICU to come from a pinned
vendored install prefix. CMake must not discover system ICU for the default
path, and the previous self-contained builtin backend is no longer built.

## Covered Behavior

- `include/tokenizers_cpp/unicode_backend.hpp` defines the backend interface for:
  - backend identity and full-ICU capability checks
  - Unicode category helpers for letter, number, mark, whitespace, control, and
    punctuation
  - UTF-8 lowercase, including span metadata for runtime offset projection
  - UTF-8 normalization with byte-span metadata
  - UTF-8 regex match offsets for components that need backend regex behavior
- `src/unicode_backend_icu.cpp` is compiled by default and uses ICU APIs for
  category, string-level lowercase with edits, normalization, and regex
  behavior.
- CMake exposes:
  - `TOKENIZERS_CPP_ICU_VENDOR`
  - `TOKENIZERS_CPP_ICU_SHARED`
  - `TOKENIZERS_CPP_ICU_ROOT`
- `TOKENIZERS_CPP_USE_ICU=OFF` is retired. The supported build always uses the
  vendored ICU backend.
- These options are CMake cache variables. The build no longer auto-imports
  same-named environment variables; pass `-D...=...` when overriding defaults.
- The ICU CMake path searches only under `TOKENIZERS_CPP_ICU_ROOT` and rejects
  `.so`, `.dylib`, and `.dll` libraries when `TOKENIZERS_CPP_ICU_SHARED=OFF`.
- `scripts/dev/build_vendored_icu4c.sh` records the intended static vendored
  build flow using ICU's `runConfigureICU`, `--enable-static`,
  `--disable-shared`, and `--with-data-packaging=archive`.
- Current local ICU4C source/install baseline:
  - source archive: `third_party/icu4c-78.3/icu4c-78.3-sources.tgz`
  - data archive: `third_party/icu4c-78.3/icu4c-78.3-data.zip`
  - checksums: `third_party/icu4c-78.3/SHASUM512.txt`
  - upstream tag: `release-78.3`
  - install prefix: `third_party/icu4c-install`
  - libraries: static `libicu*.a`
  - data archive: `share/icu/78.3/icudt78l.dat`
- The ICU build helper prepends the `uv`-managed Python executable directory to
  `PATH` before running ICU configure so data rules can be generated without
  relying on system Python.

## Current C++ Coverage

`tests/parity/unicode_backend_test.cpp` verifies the backend contract for
representative accepted-fixture categories, context-sensitive lowercase, edit
span mapping, and NFD byte-span shape.
The test intentionally does not claim full Unicode parity until component-level
ICU migrations add exact upstream fixtures.

When ICU is enabled on Linux with `TOKENIZERS_CPP_ICU_SHARED=OFF`,
`tokenizers_cpp_no_shared_icu_audit` runs `cmake/CheckNoSharedIcu.cmake`
against the Unicode backend smoke binary and fails if `ldd` reports a shared
`libicu*.so` runtime dependency or an unresolved dependency.

## Known Gaps

- The tokenizer runtime is being migrated component by component. The Llama
  `Split` pre-tokenizer now routes letter/number classification through this
  backend, and `BertNormalizer` routes whitespace/control cleanup, lowercase,
  NFD accent decomposition, and nonspacing-mark stripping through it.
  `BertPreTokenizer` punctuation splitting now also routes through this
  backend. Generic serialized `Split` regex matching now routes through ICU in
  the default build, and serialized `Replace` regex normalizers reuse the same
  backend regex match boundary while keeping replacement offset projection in
  the normalizer layer.
- ICU source and data release inputs are vendored as upstream release archives
  rather than as a git submodule. `scripts/dev/build_vendored_icu4c.sh` extracts
  them into `build/` and installs the static ICU4C prefix into
  `third_party/icu4c-install`.
- ICU normalization offset projection currently returns conservative spans for
  normalized output. ICU lowercase uses `CaseMap` edits for source span
  projection.
- ICU regex replacement is wired for serialized `Replace` normalizers. Broader
  regex-backed components should continue to reuse the backend regex boundary
  instead of calling ICU directly from tokenizer runtime code.

## Verification

When a vendored ICU install is present at the default prefix, the supported
smoke path uses the default options:

- `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-icu -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
- `cmake --build projects/tokenizers.cpp/build-icu`
- `ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure`

The default ICU Linux test count is currently 26/26 because it includes the
shared-ICU dependency audit.
