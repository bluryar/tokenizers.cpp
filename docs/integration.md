# tokenizers.cpp Integration Guide

Date: 2026-05-07

## Build Modes

`tokenizers.cpp` supports two downstream C++ consumption modes.

Source-tree consumption:

```cmake
set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/tokenizers.cpp)

target_link_libraries(your_target PRIVATE tokenizers_cpp::tokenizers_cpp)
```

Installed-package consumption:

```sh
cmake -S projects/tokenizers.cpp -B build-tokenizers \
  -DTOKENIZERS_CPP_BUILD_TESTS=ON \
  -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build build-tokenizers
cmake --install build-tokenizers --prefix /path/to/tokenizers-cpp-prefix
```

```cmake
list(PREPEND CMAKE_PREFIX_PATH "/path/to/tokenizers-cpp-prefix")
find_package(tokenizers_cpp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE tokenizers_cpp::tokenizers_cpp)
```

The installed package carries public headers, the vendored nlohmann JSON
single-header copy, and the vendored ICU static libraries required by the
exported static target. The package config recreates imported ICU targets from
the install prefix and does not discover system ICU by default.

## Minimal Runtime Use

```cpp
#include "tokenizers_cpp/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

int main() {
  const auto tokenizer =
      tokenizers_cpp::Tokenizer::from_file("tokenizer.json");

  const auto encoding = tokenizer.encode("hello world");
  const std::vector<std::uint32_t> ids = encoding.ids;
  const std::string text = tokenizer.decode(ids);

  return text.empty() ? 1 : 0;
}
```

## Public API Notes

- `encode` and `encode_pair` return byte offsets.
- Use the `*_char_offsets` overloads for Unicode scalar offsets.
- Pre-tokenized offsets are relative to each input word.
- Batch encode/decode preserves input order.
- `DecodeStream::has_pending()` reports incomplete byte fallback state.
- There is no `DecodeStream::flush()` or `finalize()`; reset by discarding the
  stream and creating a new one.

See `docs/api-stability.md` for the stable public surface.

## Examples

The `examples/` directory contains self-contained programs that create temporary
tokenizer JSON files:

- `basic_encode_decode.cpp`
- `batch_and_padding.cpp`
- `stream_decode.cpp`

When building `tokenizers.cpp` as the top-level project,
`TOKENIZERS_CPP_BUILD_EXAMPLES` defaults to `ON`; when consumed as a subproject,
set it explicitly if examples are desired.

## Runtime Boundaries

`tokenizers.cpp` is a native C++ inference library. It does not require Rust,
Python, network access, HTTP/from-pretrained loading, trainers, or wrapper
bindings at runtime.

The default supported build uses vendored/static ICU4C and includes a Linux
CTest audit that rejects accidental shared `libicu*.so` linkage.
