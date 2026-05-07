# R5 C++ Consumer Readiness

Date: 2026-05-07

## Goal

Make the completed tokenizer-centered inference runtime comfortable to consume
from downstream C++ and GGML projects.

## R5-C1 Install And Consumer Smoke

Status: completed

This slice adds the first consumer-facing build surface:

- `tokenizers_cpp::tokenizers_cpp` remains the public CMake target.
- Source-tree consumers can use `add_subdirectory`.
- Installed consumers can use `find_package(tokenizers_cpp CONFIG REQUIRED)`.
- The install tree carries public headers, the vendored nlohmann header, and
  the vendored ICU static libraries needed by the exported static target.
- The package config recreates private ICU imported targets from the install
  prefix instead of discovering system ICU.
- CTest contains out-of-tree consumer smokes for both `add_subdirectory` and
  installed-package consumption.

## Consumer Verification

The R5-C1 consumer smoke is intentionally tiny. It loads the committed
WordLevel fixture and verifies:

- `Tokenizer::from_file`
- `encode`
- `encode_pair`
- `encode_batch`
- `decode`
- `decode_batch`
- vocab lookup
- no shared ICU dependency on Linux

Latest local result:

- CMake configure passed for `build-icu`.
- C++ build passed.
- CTest passed: 28/28.
- Fixture generator check passed.
- Both consumer smokes passed.

## R5-C2 Public Header Cleanup

Status: completed

This slice hides tokenizer implementation state from the public header:

- `Tokenizer` now stores internal model, normalizer, pre-tokenizer,
  post-processor, decoder, truncation, padding, and vocabulary state behind a
  private implementation pointer.
- `include/tokenizers_cpp/tokenizer.hpp` no longer exposes internal
  `detail::*` component config structs.
- The public header keeps only the consumer-facing API types and the narrow
  stability comments needed by downstream users.
- API stability notes are recorded in `docs/api-stability.md`.

Latest local result:

- C++ build passed.
- CTest passed: 28/28.
- Fixture generator check passed.
- Both source-tree and installed-package consumer smokes still passed.

## R5-C3 Examples And Integration Guide

Status: completed

This slice adds downstream-facing usage material:

- `examples/basic_encode_decode.cpp`
- `examples/batch_and_padding.cpp`
- `examples/stream_decode.cpp`
- `docs/integration.md`
- `README.md`

The examples are self-contained and create temporary tokenizer JSON files. They
do not depend on the HF test-data checkout, Rust, Python, or network access at
runtime. In top-level builds, examples default to enabled and run as CTest
smokes; subproject consumers can leave `TOKENIZERS_CPP_BUILD_EXAMPLES=OFF`.

Latest local result:

- C++ build passed.
- CTest passed: 31/31.
- Fixture generator check passed.
- Source-tree and installed-package consumer smokes still passed.

## Remaining R5 Work

- Optional real-tokenizer consumer smoke once a specific GGML/C++ model project
  chooses its tokenizer JSON fixtures.
