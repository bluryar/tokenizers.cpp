# Open-Source Release Checklist

Use this checklist before publishing `tokenizers.cpp` as a standalone
repository.

## Repository Shape

- [ ] `third_party/tokenizers` is committed as a git submodule gitlink.
- [ ] `third_party/icu4c-78.3` contains the pinned ICU source/data release
  archives and checksum file.
- [ ] `.gitmodules` contains only public upstream URLs.
- [ ] `third_party/icu4c-install` is not committed.
- [ ] build directories such as `build/` and `build-*` are not committed.
- [ ] local HF test data under `hf-internal-testing/` is not committed.

Because `projects/tokenizers.cpp` currently lives inside the wider `ggbond`
workspace, avoid running root-level `git submodule add` for these paths. When
publishing the standalone repository, initialize the repository with
`projects/tokenizers.cpp` as its root and record the existing nested checkouts
as submodule gitlinks there.

## Parent Workspace Integration

After the standalone public repository exists, `ggbond` can consume the whole
`projects/tokenizers.cpp` project as a parent-level submodule. Do this only
after the public remote URL is known.

Expected shape:

- `ggbond/.gitmodules` owns `projects/tokenizers.cpp`.
- `projects/tokenizers.cpp/.gitmodules` owns nested upstream references:
  `third_party/tokenizers`.
- Consumers initialize recursively:

```sh
git submodule update --init --recursive projects/tokenizers.cpp
```

Do not publish `third_party/icu4c-install`, build directories, or local HF test
data under `hf-internal-testing/` through either repository level.

## Licensing And Notices

- [ ] `LICENSE` is present.
- [ ] `NOTICE` is present.
- [ ] `THIRD_PARTY_NOTICES.md` lists Hugging Face tokenizers, ICU4C,
  nlohmann/json, and aho_corasick.hpp.
- [ ] README license text matches the selected project license.

## Verification

Run the self-contained build:

```sh
git submodule update --init --recursive
scripts/dev/build_vendored_icu4c.sh
cmake -S . -B build-icu -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build build-icu
ctest --test-dir build-icu --output-on-failure
```

Run full parity when the local ignored HF test-data checkout is available:

```sh
cmake -S . -B build-icu \
  -DTOKENIZERS_CPP_BUILD_TESTS=ON \
  -DTOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS=ON \
  -DTOKENIZERS_CPP_HF_TEST_DATA_DIR=/path/to/tokenizers.cpp/hf-internal-testing/tokenizers-test-data \
  -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build build-icu
ctest --test-dir build-icu --output-on-failure
```

Development-only fixture validation:

```sh
uv run --no-project --script scripts/dev/generate_parity_fixtures.py --check
```

## Expected Publish Statement

The precise milestone statement is:

> Native C++ tokenizer-centered core inference API translation is complete.
> Remaining work is hardening, edge parity, and optional lower-level public
> surface expansion.

Do not describe the project as a complete Rust crate replacement. Trainers,
HTTP/from-pretrained, Python/Node/WASM wrappers, Rust FFI, and Unigram sampling
remain out of scope.
