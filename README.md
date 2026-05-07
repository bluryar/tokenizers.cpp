# tokenizers.cpp

Native C++ inference-only port of the Hugging Face `tokenizers` runtime surface
needed by downstream GGML/C++ projects.

This is not a Rust FFI wrapper. Runtime use does not require Rust, Python,
network access, trainers, HTTP/from-pretrained loading, or wrapper bindings.

## Repository Setup

This repository expects upstream references as git submodules:

```sh
git submodule update --init --recursive
```

`third_party/tokenizers` is the read-only Hugging Face Rust reference used for
development parity. `third_party/icu` is the pinned ICU4C source checkout used
to produce a static, project-owned ICU install.

The generated ICU install prefix is intentionally not committed:

```sh
scripts/dev/build_vendored_icu4c.sh
```

That script installs ICU under `third_party/icu4c-install`, which is the default
`TOKENIZERS_CPP_ICU_ROOT`.

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
