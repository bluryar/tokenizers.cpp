# tokenizers.cpp API Stability

Date: 2026-05-07

## Stable Public Surface

The R5 consumer-facing surface is tokenizer-centered:

- `tokenizers_cpp::Tokenizer`
- `tokenizers_cpp::Encoding`
- `tokenizers_cpp::Offset`
- `tokenizers_cpp::DecodeStream`
- `tokenizers_cpp::AddedToken`
- `tokenizers_cpp::BpeOptions`

The public CMake target is:

- `tokenizers_cpp::tokenizers_cpp`

Supported consumer entry points:

- source-tree consumption with `add_subdirectory`
- installed-package consumption with `find_package(tokenizers_cpp CONFIG REQUIRED)`

## Header Boundary

`include/tokenizers_cpp/tokenizer.hpp` now uses a private implementation pointer
for tokenizer state. Internal component config structs, merge tables, decoder
chains, normalizer state, and padding/truncation config are implementation
details in `src/tokenizer.cpp`.

Downstream code should not depend on:

- `detail::*` component config types
- concrete model internals
- BPE merge/cache state
- private tokenizer mutation beyond the narrow public methods already declared
  in `Tokenizer`

## Offset Boundary

`encode` and `encode_pair` return byte offsets, matching the default Hugging
Face tokenizer inference behavior used by the parity tests.

Use the `*_char_offsets` overloads when downstream code needs Unicode scalar
index offsets. Pre-tokenized overload offsets are relative to each input word.

## Stream Decode Boundary

`DecodeStream::step(id)` returns output only when a complete decodable chunk is
available. `DecodeStream::has_pending()` reports incomplete byte fallback state.
There is intentionally no public `flush()` or `finalize()` method; callers reset
state by discarding the stream and creating a new one.

## Intentionally Private For Now

- standalone public `models::BPE`
- BPE builder/cache controls
- broad Rust-style component builders
- trainers, training APIs, HTTP/from-pretrained, and wrapper APIs

Those boundaries are governed by `docs/adr/ADR-0004-bpe-public-surface.md` and
the core inference completion statement.
