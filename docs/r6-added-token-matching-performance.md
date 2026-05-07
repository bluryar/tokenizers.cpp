# R6 Added-Token Matching Performance

Date: 2026-05-07

## Status

R6 added-token matching performance hardening is complete.

The public API remains tokenizer-centered. The optimized matcher is a private
implementation detail behind `Tokenizer::Impl`; no third-party trie type leaks
into public headers.

## Decision

The added-token extraction hot path now uses a cached Aho-Corasick matcher when
the tokenizer has enough non-empty added-token patterns to justify trie setup.
Small added-token sets continue to use the simple legacy scan.

The matcher only collects byte-offset candidates. The C++ tokenizer still owns
the semantic selection rules:

- leftmost-longest candidate choice
- stable token-index tie break
- empty added-token skip
- `single_word` boundary rejection
- `lstrip` / `rstrip` span expansion
- UTF-8 byte offsets
- search-cursor advance after rejected `single_word` candidates

This mirrors the Rust upstream design direction of using multi-pattern
matching for added tokens while preserving this project's accepted native C++
runtime surface.

## Dependency

R6 vendors a small header-only Aho-Corasick implementation:

- Source: `https://github.com/cjgdev/aho_corasick`
- Imported header:
  `third_party/aho_corasick/aho_corasick/aho_corasick.hpp`
- License: MIT License
- Visibility: private include path for `tokenizers_cpp`; benchmark-only direct
  include for the microbenchmark target

The dependency is checked in. CMake does not fetch it from the network.

## Tests

R6 adds parity coverage for:

- overlapping added tokens such as `<m>`, `<mask>`, and `<mask>ing`
- `single_word` rejection followed by a later valid occurrence
- `lstrip` / `rstrip` with overlapping matcher candidates
- UTF-8 byte offsets through added-token extraction
- empty added tokens not entering vocab or the matcher

## Benchmark

The optional benchmark target is:

```sh
cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-bench \
  -DTOKENIZERS_CPP_BUILD_TESTS=OFF \
  -DTOKENIZERS_CPP_BUILD_EXAMPLES=OFF \
  -DTOKENIZERS_CPP_BUILD_BENCHMARKS=ON \
  -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build projects/tokenizers.cpp/build-bench \
  --target tokenizers_cpp_added_token_matcher_benchmark
projects/tokenizers.cpp/build-bench/tokenizers_cpp_added_token_matcher_benchmark
```

Current local benchmark output:

```text
tokens=16 repeats=64 legacy_ms=8.04633 trie_ms=5.69242 speedup=1.41352x
tokens=128 repeats=128 legacy_ms=521.398 trie_ms=11.4452 speedup=45.5561x
tokens=1024 repeats=256 legacy_ms=22297.9 trie_ms=22.1642 speedup=1006.03x
```

The benchmark is a performance report, not a CTest assertion.
