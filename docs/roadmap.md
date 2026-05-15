# tokenizers.cpp Roadmap

## R0. Bootstrap And Agent Harness

Status: completed

- [x] Create `projects/tokenizers.cpp`
- [x] Clone upstream `huggingface/tokenizers` as read-only local reference
- [x] Add local `AGENTS.md`
- [x] Add local `.codex/config.toml`
- [x] Add local `.codex/agents/*.toml`
- [x] Add `docs/vision.md`, `docs/roadmap.md`, `docs/status.md`
- [x] Add first ADR and parity specs
- [x] Add minimal CMake/C++ API scaffold and smoke test
- [x] Add dev fixture generator backed by upstream Rust reference
- [x] Add first generated C++ parity fixture test

## R1. Tokenizer JSON And Encoding Core

Status: completed

- Implement full tokenizer JSON deserialization for the inference graph.
- Preserve `Encoding` fields: ids, type ids, tokens, offsets, word ids,
  special-token masks, attention masks, and overflowing encodings.
- Add parity tests for `serialization.rs`, local tokenizer JSON loading, and
  added-token metadata.

## R2. Runtime Components

Status: completed for core inference; hardening follow-ups tracked in R4

- Port normalizers, pre-tokenizers, post-processors, decoders, truncation, and
  padding in dependency order.
- Add focused parity tests for offsets, Unicode handling, byte-level behavior,
  template processing, and decode behavior.
- Current completed slices include ByteLevel offsets, BertProcessing, direct
  serialized TemplateProcessing pieces, RobertaProcessing, post-processor
  Sequence for the covered ByteLevel + TemplateProcessing shape, Metaspace
  decode, ByteLevel stream decode, Llama `Split + ByteLevel(use_regex=false)`
  encode breadth for the accepted contraction/newline/Unicode category
  fixtures, direct serialized `Split` pre-tokenizer runtime for String and
  ICU Regex patterns, serialized `Replace` normalizer runtime for String and
  ICU Regex patterns, generic `Split`/`Replace` ICU regex composition before
  ByteLevel/GPT-2 BPE, deterministic BPE config breadth including dropout
  `null`/`0.0`/`1.0` boundaries, raw non-ByteLevel BPE newline byte fallback,
  public raw BPE `from_bpe_files`, merge-token OOV validation, stochastic
  dropout shape invariants, no-cross-instance cache behavior, the stream
  added-BPE runtime mutation path, single/pair
  truncation with stride overflowing, batch padding over main and overflowing
  encodings including direct pre-tokenized pair and pre-tokenized
  single/pair batch overloads, ordered batch encode/decode execution,
  explicit char-offset encode APIs,
  generalized serialized `TemplateProcessing` coverage for multi-id special
  tokens, non-default type ids, pair templates that reorder A/B, skipped
  special pieces with sequence type-id overrides, truncation accounting, and
  nested pair-overflow cross-products,
  decoder `Sequence` composition for the covered ByteFallback/Fuse,
  CTC/Metaspace, CTC/Fuse cleanup and custom-config, Strip/Replace/Fuse,
  ICU Regex Replace, and BPEDecoder fixtures,
  ICU-backed `NFC`/`NFD`/`NFKC`/`NFKD`, upstream `Nmt`, serialized `Prepend`,
  Unicode-whitespace `Strip`, ALBERT/SentencePiece string-level `Lowercase`,
  and the self-contained ALBERT `Precompiled` charsmap path covering
  zero-width joiner/non-joiner/space splitting, control separator
  mapping/deletion, record-separator deletion, and expansion-followed-by-removal
  projection,
  real BERT, RoBERTa, GPT-style, ALBERT, and Llama tokenizer JSON smoke paths
  that compose model inference, post-processing, truncation, overflowing, fixed
  padding, ordered batch/pair-batch, char-offset, and pre-tokenized output
  where covered, including ALBERT overflow/padding/batch/decode-batch, BERT
  char-offset APIs plus real-JSON pair-batch/decode-batch, RoBERTa real-JSON
  pair-batch, Llama real-JSON batch/pair-batch/decode-batch, and GPT-style
  ordered batch/pair-batch/decode-batch plus pre-tokenized single/pair batch
  output through `Sequence(ByteLevel, TemplateProcessing)`,
  plus the upstream documentation BERT pipeline over real `bert-wiki.json`
  with direct `Whitespace` pre-tokenization and runtime WordPiece decoder
  replacement, and the upstream documentation generic pipeline over real
  `tokenizer-wiki.json` with `Sequence(NFD, StripAccents)`,
  `Sequence(Whitespace, Digits)`, TemplateProcessing, and BPE decode, plus
  the upstream quicktour inference path covering initial encode, batch
  execution, and BatchLongest padding, plus
  `documentation.rs::load_tokenizer` over real `roberta.json` encode/decode,
  plus
  `documentation.rs::streaming_tokenizer` over RoBERTa ByteLevel, ALBERT
  Metaspace, and ByteFallback partial-byte streaming, plus a consolidated
  real-tokenizer smoke suite over GPT-style, RoBERTa, BERT, ALBERT, and Llama
  JSON families.
- Unicode/regex runtime coverage now routes through the project-owned ICU
  backend where accepted by core inference parity. `ADR-0003` accepts
  vendored/static ICU4C as the default Unicode backend while continuing to
  reject default system ICU dynamic-link dependency.
- The initial Unicode backend infrastructure is now in place: vendored ICU4C
  default backend, static-build helper script, backend smoke coverage, and a
  Linux shared-ICU dependency audit.

## R3. Model Inference Parity

Status: completed for core inference; hardening follow-ups tracked in R4

- Port BPE, WordPiece, WordLevel, and Unigram inference.
- Exclude trainers and training tests unless scope is explicitly revised.
- Validate real GPT/BERT/Roberta/SentencePiece-style tokenizer JSON fixtures.

## R4. Hardening And Public Surface Review

Status: active/current

- Keep the completed tokenizer-centered inference API stable.
- R4-H1 minimal hardening is frozen in
  `docs/r4-h1-minimal-hardening-freeze.md`.
- R4-H2 BPE public surface review is frozen in
  `docs/r4-h2-bpe-public-surface-review.md`; `ADR-0004` keeps BPE access
  tokenizer-centered through `Tokenizer::from_file` and
  `Tokenizer::from_bpe_files`.
- Expand only the edge cases that are needed by downstream consumers:
  BPE implementation hardening behind the accepted public surface and rare ICU
  offset projection cases.
  R4-H1 stream mixed-decoder boundaries are pinned for ByteFallback/Fuse,
  Metaspace/Fuse, ByteLevel/Fuse, and the no-finalize reset policy. R4-H1 also
  broadens ALBERT `Precompiled` charsmap validation for no-break/BOM
  boundaries, chained zero-width deletion, compatibility expansion, and
  expansion followed by joiner splitting.
- Lower-level model/component APIs remain outside the public C++ surface unless
  a downstream integration re-opens the accepted tokenizer-centered decision.

## R5. C++ Consumer Readiness

Status: active/current

- Stabilize the public C++ API for downstream GGML/C++ inference projects.
- Add install/export targets only after runtime parity is established.
- Document unsupported upstream features and exact parity gaps.
- R5-C1 completed CMake install/export support plus out-of-tree consumer
  smoke tests for both `add_subdirectory` and installed-package usage. See
  `docs/r5-consumer-readiness.md`.
- R5-C2 completed public header cleanup and API stability notes. Tokenizer
  implementation state is hidden behind PIMPL, and the consumer-facing boundary
  is recorded in `docs/api-stability.md`.
- R5-C3 completed small examples and downstream integration guide:
  self-contained encode/decode, batch/padding, and stream decode examples,
  `docs/integration.md`, and a project `README.md`.
- R5-C4 initialized open-source readiness: standalone repository submodule
  metadata for `third_party/tokenizers`, vendored ICU release archives, release
  hygiene files, third-party notices, an open-source release checklist, and
  optional gating for parity tests that require local ignored HF test data.
- Remaining R5 work is optional model-specific real-tokenizer consumer smoke
  once a downstream GGML/C++ project chooses concrete tokenizer JSON fixtures.

## R6. Performance Hardening

Status: active/current

- Keep performance work behind the tokenizer-centered public API.
- R6 added-token matching performance hardening is frozen in
  `docs/r6-added-token-matching-performance.md`.
- R6 token-to-id map performance hardening is frozen in
  `docs/r6-token-id-map-performance.md`.
- R6 BPE cache performance hardening is frozen in
  `docs/r6-bpe-cache-performance.md`.
- R6 BPE merge heap performance hardening is frozen in
  `docs/r6-bpe-heap-performance.md`.
- R6 Unigram trie/cache performance hardening is frozen in
  `docs/r6-unigram-cache-performance.md`.
- R6 hot-path follow-up performance hardening is recorded in
  `docs/r6-hot-path-followup-performance.md`.
- R6 performance measurement is recorded in
  `docs/r6-performance-measurement.md`.
- Added-token extraction now uses a private cached Aho-Corasick matcher for
  large non-empty added-token sets and keeps the simple scan for small sets.
- The matcher is candidate collection only. C++ runtime logic still owns
  leftmost-longest selection, `single_word`, `lstrip`/`rstrip`, empty-token
  skip, UTF-8 byte offsets, and rejected-candidate cursor behavior.
- Optional microbenchmarks are gated by
  `TOKENIZERS_CPP_BUILD_BENCHMARKS=ON` and are not CTest assertions.
- Encode and lookup hot paths now reuse a private persistent `token_to_id_` map
  instead of rebuilding a full vocab map per call.
- Deterministic BPE encode paths now use a private thread-local cache for
  repeated normalized pieces while preserving per-input offset projection.
- Deterministic BPE merge selection now uses a private priority queue instead
  of repeated full pair rescans; stochastic dropout keeps the previous linear
  scan.
- Deterministic Unigram encode paths now use a private vocab trie plus
  thread-local repeated-piece cache while preserving tokenizer-instance
  separation and per-input offset projection.
- Batch encode/decode avoids thread launch for small workloads, ICU regex
  matching reuses compiled expressions per thread, WordPiece greedy matching
  uses private initial/continuation tries, and common unknown/byte-fallback ids
  are cached behind the existing tokenizer runtime.
- The R6 benchmark matrix now includes real GPT-style, RoBERTa, BERT, ALBERT,
  and Llama tokenizer JSON cases when local HF test data exists, while keeping
  self-contained synthetic cases available for open-source clones without that
  data.
