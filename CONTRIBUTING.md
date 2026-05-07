# Contributing

Thanks for helping harden `tokenizers.cpp`.

## Scope

This project is a native C++ inference-only port of the tokenizer runtime
surface needed by downstream GGML/C++ projects. Please keep changes within that
scope unless an issue explicitly revises it.

In scope:

- tokenizer JSON loading
- encode/decode inference behavior
- Unicode normalization, pre-tokenization, post-processing, decoding
- model inference for BPE, WordPiece, WordLevel, and Unigram
- parity fixtures and C++ tests
- CMake packaging and consumer integration

Out of scope by default:

- trainers and training tests
- HTTP/from-pretrained loading
- Python, Node, or WASM wrappers
- Rust FFI wrappers
- patching `third_party/tokenizers`

## Development Setup

```sh
git submodule update --init --recursive
scripts/dev/build_vendored_icu4c.sh
cmake -S . -B build-icu -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build build-icu
ctest --test-dir build-icu --output-on-failure
```

Full parity tests that use real tokenizer JSON fixtures require
`TOKENIZERS_CPP_HF_TEST_DATA_DIR` to point at a local
`hf-internal-testing/tokenizers-test-data` checkout under this project root.
When that directory is not available, CMake skips those tests by default.

## Change Guidelines

- Keep the public API tokenizer-centered unless an accepted ADR revises that
  boundary.
- Prefer focused parity tests over broad rewrites.
- Do not modify files under `third_party/tokenizers` or `third_party/icu` as
  part of normal development.
- Keep generated build output, ICU install prefixes, and downloaded test data
  out of git.
