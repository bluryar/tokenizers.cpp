# Split Pre-Tokenizer Runtime Boundary

## Scope

This slice broadens serialized `Split` pre-tokenizer runtime support beyond the
previous Llama-recognized regex path.

Covered behavior:

- Direct top-level `pre_tokenizer: {"type":"Split"}` parsing.
- `Sequence` children containing `Split`, preserving the existing Llama
  `Split + ByteLevel(use_regex=false)` path.
- Direct top-level `Digits` parsing with both contiguous-number and
  individual-digit modes.
- `Sequence(Whitespace, Digits(individual_digits=true))` execution order for
  the accepted documentation pipeline shape.
- `pattern: {"String": ...}` literal matching without requiring ICU.
- `pattern: {"Regex": ...}` matching through the shared Unicode backend. The
  default vendored ICU build uses ICU regex.
- `invert=true` by flipping match flags before delimiter behavior is applied.
- `SplitDelimiterBehavior` core modes: `Removed`, `Isolated`,
  `MergedWithPrevious`, `MergedWithNext`, and `Contiguous`.
- WordLevel encode-side runtime coverage for ids, tokens, offsets, word ids,
  type ids, special-token masks, and attention masks.
- Default-ICU supplementary-plane regex offset coverage for `\p{P}+`.
- ByteLevel/BPE encode-side composition for the accepted generic ICU regex
  fixture: `Sequence(Split(Regex("\\p{P}+"), Isolated),
  ByteLevel(use_regex=false))` over GPT-2 BPE data.
- Non-ByteLevel BPE encode-side composition for the accepted
  `Sequence(Whitespace, Digits)` documentation fixture over
  `tokenizer-wiki.json`.

This remains inference-only: no Rust FFI, no runtime shell-out, no training,
no HTTP/from-pretrained loading, and no edits under `third_party/tokenizers`.

## Fixtures

`tests/parity/split_pre_tokenizer_test.cpp` covers:

- `pre_tokenizers::split::tests::regex_string` shape:
  `Regex("\\s+")` and `String(" ")` split the same whitespace boundary class
  for the accepted WordLevel fixture.
- `pre_tokenizers::split::tests::invert` shape:
  `String("Hello")`, `Removed`, `invert=true`.
- `pre_tokenizers::split::tests::basic` delimiter behavior shape:
  string-pattern `"-"` over `The-final--countdown` for all five delimiter
  modes.
- Default-ICU generic regex category shape:
  `Regex("\\p{P}+")`, `Isolated`, over `hello𐄀world`, checking byte offsets
  across a supplementary-plane punctuation match.
- `tests/parity/byte_level_bpe_test.cpp` adds the same generic ICU regex
  `Split` shape before ByteLevel/BPE, checking Rust-derived ids, tokens,
  offsets, word ids, masks, and ByteLevel decode.
- `tests/parity/split_pre_tokenizer_test.cpp` covers upstream
  `pre_tokenizers::digits::{numbers,individual_digits}` shape and
  `Sequence(Whitespace, Digits(individual_digits=true))` over `Call 911!`.
- `tests/parity/documentation_pipeline_test.cpp` covers the tokenizer-level
  documentation pipeline with real `tokenizer-wiki.json`.

## Known Gaps

- Generic regex execution is provided by the default ICU backend.
- Regex replacement normalizers now reuse the same backend regex match boundary;
  see `replace-normalizer-runtime-boundary.md`.
- Full Rust regex syntax parity is not claimed. ICU regex is the accepted
  project-owned default backend, so differences between Rust `regex` and ICU
  regex must be handled fixture by fixture.
- ByteLevel generic `Split` regex composition is covered by one accepted
  supplementary-plane punctuation fixture. A broad arbitrary-regex ByteLevel
  matrix remains a follow-up.

## Verification

- `ctest --test-dir build-icu -R tokenizers_cpp_split_pre_tokenizer_test --output-on-failure`
- Upstream reference:
  `cargo test pre_tokenizers::split::tests`
