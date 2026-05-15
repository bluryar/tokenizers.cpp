# tokenizers.cpp

Native C++ inference-only port of the Hugging Face `tokenizers` runtime surface
needed by downstream GGML/C++ projects.

This is not a Rust FFI wrapper. Runtime use does not require Rust, Python,
network access, trainers, HTTP/from-pretrained loading, or wrapper bindings.

## Repository Setup

This repository expects the Hugging Face Rust tokenizer reference as a git
submodule:

```sh
git submodule update --init --recursive
```

`third_party/tokenizers` is the read-only Hugging Face Rust reference used for
development parity.

ICU4C is vendored as pinned upstream release archives, not as a submodule:

- `third_party/icu4c-78.3/icu4c-78.3-sources.tgz`
- `third_party/icu4c-78.3/icu4c-78.3-data.zip`
- `third_party/icu4c-78.3/SHASUM512.txt`

The generated ICU install prefix is intentionally not committed:

```sh
scripts/dev/build_vendored_icu4c.sh
```

That script verifies the release inputs exist, extracts the source archive under
`build/`, and installs ICU under `third_party/icu4c-install`, which is the
default `TOKENIZERS_CPP_ICU_ROOT`.

## Build

```sh
cmake -S . -B build-icu \
  -DTOKENIZERS_CPP_BUILD_TESTS=ON \
  -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build build-icu
ctest --test-dir build-icu --output-on-failure
```

The supported default build uses the vendored/static ICU4C install under
`third_party/icu4c-install` and includes a Linux audit that rejects accidental
shared `libicu*.so` runtime linkage.

Some parity tests use Hugging Face tokenizer JSON fixtures from the local
`hf-internal-testing/tokenizers-test-data` checkout under this project root.
Those tests are enabled by default only when `TOKENIZERS_CPP_HF_TEST_DATA_DIR`
exists. The self-contained tests, examples, install/export smoke tests, and
generated fixtures do not require that checkout.

## Unicode And Regex Backend

The default backend is vendored/static ICU4C. The upstream Rust tokenizer
runtime uses a combination of `onig`, `regex`, Unicode normalization/category
crates, and SentencePiece precompiled data, so the C++ port needs broad Unicode
normalization, lowercase/category checks, regex Unicode properties, lookahead,
and offset projection. ICU covers that full surface behind the private
`tokenizers_cpp::unicode` backend without depending on OS `libicu*.so`/`.dll`
packages at runtime.

`utf8proc` is a good future candidate for an optional lightweight Unicode
backend because it is small and covers UTF-8 normalization, case folding,
grapheme helpers, and Unicode categories. It is not a drop-in replacement for
the current default because it does not provide regex, and tokenizer parity
would still need a separate regex engine plus offset-projection glue.

RE2 may be useful later as a private optional fast path for RE2-compatible
serialized regexes, but it is not the default regex backend. It does not support
lookaround assertions, while common tokenizer patterns such as the GPT-2
ByteLevel regex require negative lookahead.

## CMake Consumption

Source-tree use:

```cmake
set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/tokenizers.cpp)
target_link_libraries(app PRIVATE tokenizers_cpp::tokenizers_cpp)
```

Installed-package use:

```cmake
find_package(tokenizers_cpp CONFIG REQUIRED)
target_link_libraries(app PRIVATE tokenizers_cpp::tokenizers_cpp)
```

See `docs/integration.md` for the full consumer guide.

## Public API

The public API is tokenizer-centered:

- `Tokenizer::from_file`
- `Tokenizer::from_bpe_files`
- raw, pair, batch, pre-tokenized, and char-offset encode APIs
- `decode`, `decode_batch`, and `decode_stream`
- `Encoding`, `Offset`, `DecodeStream`, `AddedToken`, and `BpeOptions`

Internal component configs and model details are intentionally private. See
`docs/api-stability.md`.

## Examples

Self-contained examples live under `examples/`:

- `basic_encode_decode.cpp`
- `batch_and_padding.cpp`
- `stream_decode.cpp`

These examples write temporary tokenizer JSON files and do not depend on the HF
test-data checkout.

## License

`tokenizers.cpp` is released under the Apache License 2.0. See `LICENSE`,
`NOTICE`, and `THIRD_PARTY_NOTICES.md` for upstream and vendored dependency
notices.
