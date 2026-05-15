# tokenizers.cpp Status

## Current Snapshot

Date: 2026-05-07
Phase: Core Inference API Translation Complete; R6 added-token matching
performance hardening

## Milestone Freeze

The tokenizer-centered native C++ inference translation is milestone-complete.
This means `tokenizers.cpp` can load local tokenizer JSON and run the supported
public inference API over the main real tokenizer families without Rust FFI,
runtime Rust shell-out, Python, or network loading.

Completion statement:
`docs/core-inference-api-translation-complete.md`.

R4-H1 minimal hardening freeze:
`docs/r4-h1-minimal-hardening-freeze.md`.

R4-H2 BPE public surface review:
`docs/r4-h2-bpe-public-surface-review.md` and
`docs/adr/ADR-0004-bpe-public-surface.md`.

R5 consumer readiness:
`docs/r5-consumer-readiness.md`.

R6 added-token matching performance hardening:
`docs/r6-added-token-matching-performance.md`.

R6 token-to-id map performance hardening:
`docs/r6-token-id-map-performance.md`.

R6 BPE cache performance hardening:
`docs/r6-bpe-cache-performance.md`.

R6 BPE merge heap performance hardening:
`docs/r6-bpe-heap-performance.md`.

R6 Unigram trie/cache performance hardening:
`docs/r6-unigram-cache-performance.md`.

R6 hot-path follow-up performance hardening:
`docs/r6-hot-path-followup-performance.md`.

R6 performance measurement:
`docs/r6-performance-measurement.md`.

Open-source release checklist:
`docs/open-source-checklist.md`.

The milestone is intentionally scoped to inference. Trainers, training tests,
HTTP/from-pretrained, Python/Node/WASM wrappers, Unigram sampling, and broad
internal model-builder parity remain outside the accepted core milestone.

## Upstream Reference

- Repository: `https://github.com/huggingface/tokenizers.git`
- Local path: `third_party/tokenizers`
- Focus crate: `third_party/tokenizers/tokenizers`
- Initial `main` ref checked by clone:
  `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`
- Test data source:
  `hf-internal-testing/tokenizers-test-data`
- Crate test-data link:
  `third_party/tokenizers/tokenizers/data` -> test data source
- ICU4C source/data:
  `third_party/icu4c-78.3/icu4c-78.3-sources.tgz`,
  `third_party/icu4c-78.3/icu4c-78.3-data.zip`,
  upstream tag `release-78.3`, checksums in
  `third_party/icu4c-78.3/SHASUM512.txt`
- ICU4C static install:
  `third_party/icu4c-install`, ICU `78.3`, data archive
  `share/icu/78.3/icudt78l.dat`

`third_party/tokenizers` is prepared as a git submodule for standalone
publication. ICU4C is vendored as upstream release archives rather than as a
submodule. `third_party/icu4c-install` is a generated static install prefix and
remains ignored by this project's `.gitignore`.

## Current Surface

- Local agent environment is defined under `.codex/`.
- Project docs, ADR, parity spec, and upstream test inventory are initialized.
- JSON/component boundary is accepted in
  `docs/adr/ADR-0002-json-component-boundary.md`.
- Vendored/static ICU4C is accepted and installed locally as the default
  Unicode backend under
  `docs/adr/ADR-0003-vendored-static-icu4c-unicode-backend.md`. The policy is
  to avoid system ICU `.so` / `.dll` runtime binding, not to avoid a pinned
  project-owned Unicode library.
- The Unicode backend boundary is initialized in
  `docs/specs/icu-unicode-backend.md`: the default build uses the ICU backend
  from a vendored install under `TOKENIZERS_CPP_ICU_ROOT` and rejects
  accidental shared ICU libraries when `TOKENIZERS_CPP_ICU_SHARED=OFF`.
  `TOKENIZERS_CPP_USE_ICU=OFF` is retired, and the previous builtin Unicode
  backend has been removed from the build. ICU settings are CMake cache
  variables only; the build no longer auto-imports same-named environment
  variables. `scripts/dev/build_vendored_icu4c.sh` uses the uv-managed Python
  executable for ICU data-rule generation.
- The default ICU Linux test suite includes
  `tokenizers_cpp_no_shared_icu_audit`, which fails if `ldd` reports shared
  `libicu*.so` runtime linkage or an unresolved runtime dependency.
- Open-source release hygiene is initialized: `.gitmodules` records the public
  Hugging Face tokenizers upstream URL, ICU release archives are vendored under
  `third_party/icu4c-78.3`, generated ICU install output remains
  ignored, and top-level `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, `.gitattributes`, and
  `docs/open-source-checklist.md` are present.
- R6 performance hardening adds a private cached Aho-Corasick matcher for
  added-token extraction when the non-empty added-token set is large enough to
  benefit from trie matching. Small added-token sets keep the legacy scan.
  Candidate collection is optimized, but the tokenizer still owns
  leftmost-longest selection, `single_word`, `lstrip`/`rstrip`, empty-token
  skip, and UTF-8 byte-offset behavior. The vendored header-only dependency is
  recorded in `THIRD_PARTY_NOTICES.md`.
- R6 performance hardening also keeps a private persistent `token_to_id_` map
  inside `Tokenizer::Impl`, maintained together with `id_to_token_` during
  tokenizer JSON load, post-processor special-token registration, padding token
  registration, and runtime `add_tokens`. Encode paths and `token_to_id` now
  reuse this map instead of rebuilding a full vocab lookup table per call.
- R6 BPE performance hardening adds a private thread-local BPE cache keyed by
  per-tokenizer cache id and normalized piece text. The cache stores BPE symbol
  ids plus normalized byte ranges, then projects offsets through the current
  input piece so repeated words at different original byte positions keep
  correct offsets. Cache entries are skipped for stochastic dropout and very
  long pieces, and runtime `add_tokens` invalidates the cache id.
- R6 deterministic BPE merge hardening replaces repeated full pair rescans with
  a private priority queue of adjacent merge candidates. The heap path is used
  for deterministic dropout settings only; stochastic dropout keeps the prior
  linear scan so random candidate skipping does not drift.
- R6 Unigram performance hardening builds a private byte trie for Unigram
  vocabularies and uses a private thread-local cache for repeated normalized
  pieces. Best-path inference now walks matching trie prefixes instead of
  scanning the whole vocabulary at each byte position, while final offsets are
  still projected through the current normalized piece.
- R6 follow-up hot-path hardening keeps small batch encode/decode workloads
  serial, caches compiled ICU regex objects per thread, builds private
  WordPiece initial/continuation tries, and caches common unknown/byte-fallback
  ids after tokenizer load and runtime token mutation.
- R6 optional benchmark coverage now includes direct added-token legacy/trie
  comparison and a tokenizer-level public-API runtime matrix for
  added-token-heavy encode, repeated short BPE cache behavior, deterministic
  long-piece BPE heap behavior, Unigram trie/cache behavior, WordPiece trie
  behavior, small-batch behavior, and real
  GPT-style, RoBERTa, BERT, ALBERT, and Llama tokenizer JSON cases when local
  HF test data exists. These benchmarks are gated by
  `TOKENIZERS_CPP_BUILD_BENCHMARKS=ON` and are not CTest assertions.
- HF test-data-dependent parity tests are now gated by
  `TOKENIZERS_CPP_BUILD_HF_TEST_DATA_TESTS`. The default is ON only when
  `TOKENIZERS_CPP_HF_TEST_DATA_DIR` exists, so a normal open-source clone can
  still build self-contained tests, examples, and consumer smoke tests without
  the external fixture checkout.
- Runtime migration through the Unicode backend is complete for the Llama
  `Split` pre-tokenizer's letter/number classification and the BERT
  normalizer's whitespace/control cleanup, lowercase, NFD accent decomposition,
  and nonspacing-mark stripping. The default ICU build now covers Rust-derived
  `AሴB Ⅻ3` and `Ἀ he\u{E000}llo` parity.
- C++ public inference API exists for:
  - `Tokenizer::from_file(path)`
  - `Tokenizer::from_bpe_files(vocab_path, merges_path, options={})`
  - `encode(text, add_special_tokens=true)`
  - `encode(pre_tokenized_words, add_special_tokens=true)`
  - `encode_char_offsets(text, add_special_tokens=true)`
  - `encode_char_offsets(pre_tokenized_words, add_special_tokens=true)`
  - `encode_pair(text_a, text_b, add_special_tokens=true)`
  - `encode_pair(pre_tokenized_a, pre_tokenized_b, add_special_tokens=true)`
  - `encode_pair_char_offsets(text_a, text_b, add_special_tokens=true)`
  - `encode_pair_char_offsets(pre_tokenized_a, pre_tokenized_b, add_special_tokens=true)`
  - `encode_batch(texts, add_special_tokens=true)`
  - `encode_batch(pre_tokenized_texts, add_special_tokens=true)`
  - `encode_batch_char_offsets(texts, add_special_tokens=true)`
  - `encode_batch_char_offsets(pre_tokenized_texts, add_special_tokens=true)`
  - `encode_batch_pairs(pairs, add_special_tokens=true)`
  - `encode_batch_pairs(pre_tokenized_pairs, add_special_tokens=true)`
  - `encode_batch_pairs_char_offsets(pairs, add_special_tokens=true)`
  - `encode_batch_pairs_char_offsets(pre_tokenized_pairs, add_special_tokens=true)`
  - `decode(ids, skip_special_tokens=true)`
  - `decode_batch(sequences, skip_special_tokens=true)`
  - `decode_stream(skip_special_tokens=true)` with `DecodeStream::step(id)`
    and `DecodeStream::has_pending()`
  - `token_to_id`, `id_to_token`, `get_vocab_size`
  - narrow runtime mutation APIs for the accepted stream/documentation slices:
    `add_tokens`, `with_byte_level_normalizer`, `with_split_pre_tokenizer`,
    and `with_wordpiece_decoder`
- The tokenizer-centered public inference API is milestone-complete for the
  accepted native C++ surface. Remaining work is hardening, edge parity, and
  optional public model/component surface expansion rather than the original
  scaffold bootstrap.
- The BPE public surface review is accepted in `ADR-0004`: keep BPE access
  tokenizer-centered through `Tokenizer::from_file` and
  `Tokenizer::from_bpe_files`; do not expose standalone `models::BPE`,
  `BpeBuilder`, explicit cache controls, or exact stochastic dropout controls
  unless a concrete downstream integration re-opens the ADR.
- Dev fixture generation exists in
  `scripts/dev/generate_parity_fixtures.py`; it uses upstream Rust only at
  development time and writes committed JSON fixtures for C++ tests.
- First generated fixture coverage exists for a simple WordLevel +
  WhitespaceSplit tokenizer under `tests/parity/fixtures`.
- JSON wrapper dispatch validation now runs through `Tokenizer::from_file` for
  dispatch-only model, normalizer, pre-tokenizer, post-processor, and decoder
  slots. This covers exact upstream tags such as `BertNormalizer` and
  `CharDelimiterSplit`, the exact upstream `BertProcessing` serde shape,
  rejects legacy aliases, and rejects wrong-slot types.
- nlohmann/json is vendored as a copied single header under the project include
  tree, so CMake configure/build no longer needs `FetchContent` for JSON.
- Model JSON load validation now covers realistic BPE, WordPiece, WordLevel,
  and legacy Unigram/ALBERT shapes through `Tokenizer::from_file`. This is load
  and vocab/id lookup coverage plus the narrow GPT-2 ByteLevel/BPE encode path
  needed by the first byte-level offset slice; WordPiece now also has greedy
  subword, unknown-token, and max-input-character runtime coverage. WordLevel
  now has unknown-token fallback and missing-unk failure coverage. BPE now has
  runtime coverage for unknown fallback/fusing, `ignore_merges`, continuing
  subword prefix merge derivation, end-of-word suffix lookup, ASCII byte
  fallback, heap-backed deterministic merge selection, and the public raw BPE
  file-loader surface. Unigram now has
  trie-backed deterministic best-path inference, fused unknown, byte-fallback,
  repeated-piece cache offset preservation, tokenizer-instance cache
  separation, and real `data/unigram.json` fixture coverage.
- Added-token JSON validation now preserves full metadata for `content`,
  `single_word`, `lstrip`, `rstrip`, `normalized`, and `special`, then routes
  records through upstream-style runtime id assignment: existing model ids win,
  and new ids are allocated after the current vocabulary instead of blindly
  trusting serialized ids.
- Initial added-token extraction runs before the current
  WordLevel/WhitespaceSplit-style tokenization path. Dedicated C++ coverage now
  checks id assignment, upstream all-zero encode special masks, decode skipping,
  single-word matching, leftmost-longest overlap precedence, and basic
  lstrip/rstrip span capture.
  Byte-level BPE coverage now checks upstream `lstrip_tokens` and
  `rstrip_tokens` token/offset behavior, including Unicode byte offsets and
  prefix-space interaction.
- The native runtime boundary for the byte-level slice is specified in
  `docs/specs/byte-level-bpe-runtime-boundary.md`. It covers only the
  ByteLevel pre-tokenizer/post-processor/decoder and deterministic GPT-2 BPE
  behavior needed by
  `added_tokens.rs::{lstrip_tokens,rstrip_tokens}` and
  `offsets.rs::{byte_level_basic,byte_level_unicode}`.
- C++ ByteLevel/BPE parity now also covers
  `offsets.rs::{byte_level_double_sequence,byte_level_pre_tokenized_sequence}`.
  A conservative public `encode(std::vector<std::string>, add_special_tokens)`
  overload accepts already pre-tokenized words, keeps offsets relative to each
  input word, and assigns word ids from the input word index.
  The ByteLevel/BPE path now also has a narrow generic ICU regex composition
  fixture: serialized `Replace(Regex("\\p{P}+") -> " ")` before direct
  ByteLevel/GPT-2 BPE, plus serialized
  `Sequence(Split(Regex("\\p{P}+"), Isolated), ByteLevel(use_regex=false))`.
  The same pre-tokenizer runtime now covers direct `Digits` in contiguous and
  individual modes plus the `Sequence(Whitespace, Digits)` order used by
  `documentation.rs::pipeline`.
- The durable acceptance boundary for that pair/pre-tokenized offsets slice is
  now recorded in `docs/specs/byte-level-pair-pretokenized-offsets.md`. It
  requires exact comparison of ids, tokens, offsets, word ids, type ids,
  special-token masks, and attention masks, and keeps the scope inference-only:
  no Rust FFI, runtime shell-out, training, HTTP/from-pretrained loading, or
  edits under `third_party/tokenizers`.
- C++ BERT/WordPiece parity now covers
  `offsets.rs::split_on_added_tokens_bert`. The exact acceptance vectors and
  failure modes are recorded in
  `docs/specs/split-on-added-tokens-bert.md`. The implementation covers only
  the minimum chain needed by that test plus the follow-on breadth in
  `docs/specs/wordpiece-bert-processing-breadth.md`: `BertNormalizer`,
  `BertPreTokenizer`, WordPiece greedy matching, optional WordPiece decoder load
  compatibility, and `BertProcessing` insertion for single and pair sequences.
  WordPiece encode now also applies the serialized simple normalizer sequence
  when a tokenizer uses `Sequence(NFD, Lowercase, StripAccents)` instead of a
  direct `BertNormalizer`, as pinned by the real local `bert-wiki.json` smoke.
  The direct serialized `Whitespace` pre-tokenizer now executes through the
  ICU regex backend, so the upstream documentation BERT pipeline splits
  `library.` as `library` plus `.` instead of treating it as one WordPiece
  word.
  `BertNormalizer` now routes common control/Unicode whitespace cleanup,
  NFD-based accent stripping, nonspacing-mark removal, and lowercase through
  the shared Unicode backend. The default ICU build includes a Rust-derived
  `Ἀ he\u{E000}llo` fixture covering full-category cleanup, decomposition,
  lowercasing, token ids, tokens, offsets, and decode.
  `BertPreTokenizer` now routes punctuation splitting through the shared
  Unicode backend while preserving original byte offsets. The default ICU build
  includes supplementary-plane punctuation coverage. WordPiece decoder JSON
  now preserves `prefix`/`cleanup` behavior and applies upstream cleanup rules
  for punctuation and common contractions. The same BERT acceptance surface now
  loads real local `bert-wiki.json` with injected truncation and fixed padding
  and verifies raw/pre-tokenized single, batch, pair, and pair-batch APIs,
  including overflowing subword encodings. Explicit char-offset overloads now
  run over the same real BERT matrix and project multi-byte normalized input
  such as `Héllo` from UTF-8 byte spans to scalar offsets.
  The same test now covers `documentation.rs::pipeline_bert`: real
  `bert-wiki.json` encode fields, no-decoder WordPiece token joining, and
  runtime `with_wordpiece_decoder()` cleanup.
- C++ documentation pipeline parity now covers the inference-only runtime slice
  recorded in `docs/specs/documentation-pipeline-runtime-boundary.md`: real
  local `tokenizer-wiki.json`, serialized `Sequence(NFD, StripAccents)`,
  serialized `Sequence(Whitespace, Digits(individual_digits=true))` before
  non-ByteLevel BPE, serialized `TemplateProcessing`, exact encode fields for
  `Hello, y'all! How are you 😁 ?`, and decode with special-token skipping.
- C++ documentation quicktour parity now covers the inference-only runtime
  slice recorded in `docs/specs/documentation-quicktour-runtime-boundary.md`:
  real local `tokenizer-wiki.json` initial encode, emoji byte offsets,
  `token_to_id`, TemplateProcessing single/pair encode, ordered
  batch/batch-pair execution, and BatchLongest padding tokens plus attention
  mask.
- C++ documentation load parity now covers the inference-only runtime slice
  recorded in `docs/specs/documentation-load-runtime-boundary.md`: real local
  `roberta.json` load, `encode("This is an example", false)` ids/tokens, and
  decode roundtrip.
- C++ BPE config parity now covers the deterministic inference subset recorded
  in `docs/specs/bpe-config-runtime-boundary.md`: unknown fallback with and
  without fusing, configured missing-unk failure, `ignore_merges=true`,
  continuing-subword prefix merge output, end-of-word suffix lookup, and ASCII
  byte fallback, including raw non-ByteLevel newline byte fallback. Raw BPE
  encode now works without a ByteLevel pre-tokenizer, and C++ coverage pins
  independent tokenizers with different merge tables so future cache work cannot
  leak results across instances. Deterministic merge selection now uses a
  private heap and pins stale-candidate invalidation. BPE `dropout` is loaded
  and validated;
  `null`/`0.0` keeps deterministic merges, `1.0` skips all merges, and
  stochastic `0<p<1` behavior is covered by shape invariants instead of exact
  random output fixtures. `Tokenizer::from_bpe_files(vocab, merges, options)`
  now covers the inference subset of upstream `BPE::from_file`, including
  `#version: 0.2` merges files, options, bad merge lines, and merge-token OOV
  failures. Tokenizer JSON load also rejects merge entries that refer to
  out-of-vocabulary tokens.
- C++ Unigram parity now covers the deterministic inference subset recorded in
  `docs/specs/unigram-runtime-boundary.md`: ordered vocab/score loading,
  optimized best-path dynamic programming, fused unknown spans, byte fallback,
  and the upstream `unigram.rs::test_unigram_from_file` fixture.
- C++ SentencePiece/ALBERT parity now covers the runtime subset recorded in
  `docs/specs/sentencepiece-albert-runtime-boundary.md`: covered normalizer
  sequence behavior, ICU-backed NFD/NFKD, StripAccents, and string-level
  Lowercase,
  self-contained `Precompiled` charsmap transform for covered ALBERT cases
  including zero-width joiner/non-joiner/space splitting, control separator
  mapping/deletion, chained zero-width deletion, no-break/BOM boundaries,
  compatibility expansion, and expansion-followed-by-removal/joiner offset
  projection,
  `WhitespaceSplit + Metaspace`, TemplateProcessing `[CLS]/[SEP]` insertion,
  Metaspace decode, and real `albert-base-v1-tokenizer.json` encode/decode
  smoke vectors.
- C++ RoBERTa processing parity now covers the runtime subset recorded in
  `docs/specs/roberta-processing-runtime-boundary.md`: direct
  `RobertaProcessing` JSON load, RoBERTa single and pair special-token
  insertion, all-zero type ids, ByteLevel offset trimming from the processor,
  pair truncation accounting with four special tokens, real local
  `roberta.json` encode/decode smoke vectors, and a real `roberta.json`
  truncation + fixed-padding smoke that checks main and overflowing encodings
  after RoBERTa special-token insertion. The real RoBERTa temporary JSON smoke
  now also covers pre-tokenized single/pair and batch/pair-batch overloads,
  preserving word-relative offsets and word ids through overflowing encodings.
  Direct serialized
  `TemplateProcessing` now compiles explicit sequence/special-token pieces
  instead of relying on the older BERT/ALBERT-shaped shortcut.
- C++ post-processor Sequence parity now covers the runtime subset recorded in
  `docs/specs/post-processor-sequence-runtime-boundary.md`: nested
  `Sequence` parsing for supported children, `ByteLevel + TemplateProcessing`
  composition, special-token vocab loading from sequence children,
  Rust-derived GPT-style `tokenizer.json` encode/decode smoke vectors, and a
  GPT-style temporary JSON smoke that composes truncation, overflowing, and
  fixed padding with the same post-processor sequence. The same GPT-style
  temporary JSON path now also covers pre-tokenized single/pair and
  batch/pair-batch overloads, preserving word-relative offsets and word ids
  through overflowing encodings.
- C++ real tokenizer smoke suite now covers the downstream-facing matrix
  recorded in `docs/specs/real-tokenizer-smoke-suite.md`: GPT-style
  `Sequence(ByteLevel, TemplateProcessing)`, RoBERTa ByteLevel BPE, BERT
  WordPiece, ALBERT/SentencePiece Unigram, and Llama Split + ByteLevel BPE
  through real local tokenizer JSON load paths. The GPT-style entry now covers
  ordered batch, pair-batch, and `decode_batch` over the accepted
  `Sequence(ByteLevel, TemplateProcessing)` shape. The RoBERTa entry now covers
  ordered pair-batch over the real post-processor shape. The BERT entry now
  covers ordered pair-batch over the real TemplateProcessing pair shape and
  `decode_batch` after installing the WordPiece decoder. The ALBERT entry now
  also injects truncation and fixed padding into a temporary real JSON copy and
  checks main plus overflowing encodings, ordered batch encode, and skip/full
  `decode_batch`. The Llama entry now covers ordered batch, pair-batch, and
  `decode_batch` over the real `Split + ByteLevel` path.
- C++ Llama encode parity now covers the runtime subset recorded in
  `docs/specs/llama-split-bytelevel-runtime-boundary.md`: pre-tokenizer
  `Sequence(Split, ByteLevel(use_regex=false))`, Llama-style regex splitting,
  three-digit number chunking, contraction splits, newline/multi-space
  splitting, Unicode backend letter/number classification, real local
  `llama-3-tokenizer.json` single/pair encode smoke vectors, and a temporary
  real Llama JSON truncation + fixed-padding smoke covering single and pair
  overflowing encodings plus ordered regular, pair, pre-tokenized, and
  char-offset batch APIs.
- C++ Split pre-tokenizer parity now covers the runtime subset recorded in
  `docs/specs/split-pretokenizer-runtime-boundary.md`: direct serialized
  `Split` parsing, String patterns, ICU-backed Regex patterns in the default
  build, `invert=true`, all five `SplitDelimiterBehavior` modes, and WordLevel
  encode-side ids/tokens/offsets/word ids/masks.
- C++ Replace normalizer parity now covers the runtime subset recorded in
  `docs/specs/replace-normalizer-runtime-boundary.md`: direct and Sequence
  `Replace` parsing, String patterns, ICU-backed Regex patterns in the default
  build, replacement offset projection in the normalizer layer, and
  Unigram/SentencePiece encode-side ids/tokens/offsets/word ids/masks.
- C++ Unicode normalizer parity now covers the runtime subset recorded in
  `docs/specs/unicode-normalizer-runtime-boundary.md`: direct and Sequence
  `Strip`, `NFC`, `NFKC`, `NFD`, `NFKD`, `Nmt`, `Prepend`, `StripAccents`, and
  `Lowercase` execution through the vendored ICU backend where applicable.
  Focused coverage now pins `Sequence(Strip, NFC)` composition, `NFKC`
  compatibility expansion, upstream `Nmt` removal/space mapping, and
  `Prepend` first-span projection with original-byte offset projection; the
  local GPT-style `tokenizer.json` smoke now also exercises its serialized
  `Sequence(Strip, NFC)` normalizer before ByteLevel/BPE/post-processing.
- C++ stream decode and runtime mutation parity now covers the runtime subset
  recorded in `docs/specs/stream-decode-runtime-boundary.md`: Llama ByteLevel
  decode with added-token ids, runtime ByteLevel normalizer and Split
  pre-tokenizer replacement, runtime `add_tokens` for the upstream added-BPE
  fixture, stateful `decode_stream(false).step(id)` buffering for partial
  UTF-8 byte sequences, upstream documentation RoBERTa ByteLevel stream
  segments, ALBERT Metaspace first-token behavior, and ByteFallback
  partial-byte `None` behavior plus orphan-byte replacement when a later
  non-byte token flushes the pending run. The stream API intentionally has no
  `flush()`/`finalize()` method; `DecodeStream::has_pending()` exposes whether
  incomplete or invalid bytes remain buffered.
- C++ decoder Sequence parity now covers the runtime subset recorded in
  `docs/specs/decoder-sequence-runtime-boundary.md`: direct and nested
  decoder JSON wrappers are lowered into an ordered native decoder-step chain,
  with focused coverage for `ByteFallback`, `Fuse`, `CTC`, `Metaspace`,
  `Strip`, `Replace` String/ICU Regex, `BPEDecoder`, `ByteLevel`, and
  `WordPiece` chain semantics.
- C++ TemplateProcessing parity now covers the runtime subset recorded in
  `docs/specs/template-processing-runtime-boundary.md`: direct serialized
  piece-array templates with multi-id special tokens, non-default type ids,
  pair templates that reorder `A`/`B`, skipped special pieces while retaining
  sequence order/type-id overrides, multi-id special-token truncation
  accounting, and nested pair-overflow cross-products.
- C++ truncation parity now covers the runtime subset recorded in
  `docs/specs/truncation-runtime-boundary.md`: tokenizer JSON truncation load,
  default Right direction for older JSON, single-sequence right/left
  truncation, stride overflowing, and single-sequence special-token accounting
  before Bert/ALBERT-shaped post-processing. `OnlySecond` now fails on
  single-sequence input when truncation would require a missing second
  sequence. Pair truncation now covers `LongestFirst`, `OnlyFirst`, and
  `OnlySecond`, including pair overflow merge order and pair special-token
  accounting for the Bert/ALBERT-shaped post-processor.
- C++ padding parity now covers the runtime subset recorded in
  `docs/specs/padding-runtime-boundary.md`: tokenizer JSON padding load,
  fixed and BatchLongest strategies, left/right direction,
  `pad_to_multiple_of`, single/pair padding after post-processing, recursive
  padding of overflowing encodings, and shared BatchLongest targets across
  public single-sequence, pair, direct pre-tokenized pair, pre-tokenized
  single-sequence batch, and pre-tokenized pair batch APIs.
- C++ tokenizer-level batch execution now uses the internal boundary recorded in
  `docs/specs/batch-runtime-boundary.md`: public batch encode/decode APIs keep
  output order stable while per-item work can execute independently before the
  shared post-batch padding step.
- C++ tokenizer-level char-offset APIs now use the boundary recorded in
  `docs/specs/char-offset-runtime-boundary.md`: default encode APIs remain
  byte-offset based, while explicit char-offset overloads project UTF-8 byte
  offsets to scalar-value indexes for single, pair, pre-tokenized, and batch
  inputs.

## Verification

- C++ supported default ICU smoke passed:
  - `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-icu -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
  - `cmake --build projects/tokenizers.cpp/build-icu`
  - `ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure`
  - Result: passed, `31/31` tests including the shared-ICU dependency audit.
- Open-source no-HF-data smoke passed:
  - `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-open-source-smoke -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF -DTOKENIZERS_CPP_HF_TEST_DATA_DIR=/tmp/tokenizers-cpp-missing-hf-test-data`
  - `cmake --build projects/tokenizers.cpp/build-open-source-smoke`
  - `ctest --test-dir projects/tokenizers.cpp/build-open-source-smoke --output-on-failure`
  - Result: passed, `19/19` self-contained tests with HF test-data-dependent
    tests skipped by default.
- Upstream Rust reference attempt:
  - `cargo test --test added_tokens --test offsets --test serialization --test stream --test unigram -- --skip test_train_unigram_from_file --skip test_sample`
  - Result: passed after linking the external test-data directory into
    `third_party/tokenizers/tokenizers/data`. Coverage: added tokens, offsets,
    serialization, stream decode, and Unigram load. Upstream
    `byte_level_pre_tokenized_sequence_with_trimming` remains ignored by the
    upstream test itself.
- Upstream documentation runtime reference:
  - `cargo test --test documentation -- --skip train_tokenizer --skip quicktour_slow_train --skip train_pipeline_bert`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test documentation load_tokenizer -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test documentation pipeline -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test documentation pipeline_bert -- --exact`
  - Result: passed. Coverage: local tokenizer load, streaming tokenizer,
    quicktour, generic pipeline, and BERT pipeline. The exact `pipeline`
    spot check passed after the C++ `Sequence(Whitespace, Digits)` runtime
    fixture was added; the exact `pipeline_bert` spot check passed after the
    C++ direct `Whitespace` runtime fixture was added. The environment still
    prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo
    exits successfully.
- Upstream Digits pre-tokenizer reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::digits::tests::numbers -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::digits::tests::individual_digits -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The
    environment still prints
    `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream test inventory audit:
  - `uv run --no-project --script projects/tokenizers.cpp/scripts/dev/inventory_upstream_tests.py`
  - Result: found 41 upstream Rust tests under `tokenizers/tests`.
    `docs/specs/upstream-test-inventory.md` now lists each test explicitly,
    plus `common/mod.rs` as reference-only fixture code.
- Dev fixture generator:
  - `uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check`
  - Result: passed; committed generated fixtures are current.
- C++ parity tests:
  - `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-icu -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
  - `cmake --build projects/tokenizers.cpp/build-icu`
  - `ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure`
  - Result: passed, `31/31` tests with the default vendored ICU backend.
- C++ Unicode backend smoke:
  - `tests/parity/unicode_backend_test.cpp`
  - Result: passed. Coverage verifies representative category helpers,
    lowercase, NFD text output, regex matching, and byte-span shape on the
    default ICU backend. This is infrastructure coverage, not full upstream
    Unicode parity by itself.
- C++ serialization wrapper dispatch parity:
  - `tests/parity/json_wrapper_dispatch_test.cpp`
  - Result: passed. Coverage corresponds to
    `serialization.rs::{normalizers,processors,pretoks,decoders,models,tokenizer}`
    as C++ load acceptance/failure through `Tokenizer::from_file`, including
    the exact upstream `BertProcessing` serde shape.
- C++ real tokenizer smoke suite:
  - `tests/parity/real_tokenizer_smoke_test.cpp`
  - Result: passed. Coverage checks full `Encoding` fields and decode for the
    five core real-tokenizer families: GPT-style, RoBERTa, BERT, ALBERT, and
    Llama. GPT-style now also covers ordered batch, pair-batch, and
    `decode_batch` over the same real-JSON path. RoBERTa now also covers
    ordered pair-batch over the real post-processor shape. BERT now also covers
    ordered pair-batch over the real TemplateProcessing pair shape and
    WordPiece decoder `decode_batch`. ALBERT now also covers real-JSON
    truncation, fixed padding, overflowing encodings, ordered batch encode, and
    skip/full `decode_batch` over main and overflow outputs. Llama now also
    covers ordered batch, pair-batch, and `decode_batch` over the real
    `Split + ByteLevel` path.
- C++ added-token parity:
  - `tests/parity/added_tokens_test.cpp`
  - Result: passed. Coverage maps to the non-byte-level core of
    `added_tokens.rs::{add_tokens,single_word_tokens,overlapping_tokens}` and
    adds partial lstrip/rstrip whitespace-span checks for the current
    WordLevel path. Full byte-level lstrip/rstrip token and offset output is
    covered by `tests/parity/byte_level_bpe_test.cpp`.
- C++ byte-level BPE parity:
  - `tests/parity/byte_level_bpe_test.cpp`
  - Result: passed. Coverage corresponds to
    `added_tokens.rs::{lstrip_tokens,rstrip_tokens}` and
    `offsets.rs::{byte_level_basic,byte_level_unicode,byte_level_double_sequence,byte_level_pre_tokenized_sequence}`
    using local GPT-2 vocab/merges from
    `hf-internal-testing/tokenizers-test-data`. It also covers Rust-derived
    generic ICU regex composition for `Replace` and `Split` before
    ByteLevel/GPT-2 BPE.
- C++ serialization model load parity:
  - `tests/parity/serialization_model_load_test.cpp`
  - Result: passed. Coverage corresponds to
    `serialization.rs::{bpe_serde,wordpiece_serde,wordlevel_serde,test_deserialize_long_file}`
    by loading GPT-2 BPE vocab/merges, BERT WordPiece vocab, GPT-2 WordLevel
    vocab, and `albert-base-v1-tokenizer.json` from the local HF test-data
    checkout. Negative checks now reject BPE models with missing `merges`,
    malformed merge entries, and malformed vocab ids.
- C++ WordLevel runtime parity:
  - `tests/parity/wordlevel_test.cpp`
  - Result: passed. Coverage corresponds to
    `models/wordlevel::test_tokenize_unk` and
    `models/wordlevel::test_tokenize_missing_unk_token` for native encode
    behavior.
- C++ BPE config runtime parity:
  - `tests/parity/bpe_config_test.cpp`
  - Result: passed. Coverage corresponds to
    `models/bpe::model::{test_unk_not_fused,test_unk_get_fused,test_bpe_with_continuing_subword_prefix,test_bpe_byte_fallback,test_ignore_merges}`
    and deterministic boundaries from `test_tokenize_with_and_without_dropout`
    for the native encode path.
- C++ BERT/WordPiece runtime parity:
  - `tests/parity/bert_wordpiece_added_tokens_test.cpp`
  - Result: passed. Coverage corresponds to
    `offsets.rs::split_on_added_tokens_bert`, plus WordPiece greedy
    subword/unknown/max-character behavior, BertProcessing single/pair
    insertion, BertNormalizer cleanup/lowercase/accent behavior through the
    shared Unicode backend, and common Unicode punctuation splitting in
    BertPreTokenizer. It now also covers WordPiece decoder prefix and cleanup
    behavior, serialized simple normalizer sequences before WordPiece, and real
    local `bert-wiki.json` truncation + fixed-padding smoke over raw and
    pre-tokenized batch/pair-batch APIs, including explicit char-offset
    overloads, recursive conversion of overflowing subword encodings, direct
    `Whitespace` pre-tokenizer runtime, and `documentation.rs::pipeline_bert`.
- C++ Unigram runtime parity:
  - `tests/parity/unigram_test.cpp`
  - Result: passed. Coverage corresponds to
    `unigram.rs::test_unigram_from_file` plus deterministic best-path,
    byte-fallback behavior from `models/unigram::model`, and R6 trie/cache
    offset and instance-separation hardening.
- C++ SentencePiece/ALBERT runtime parity:
  - `tests/parity/sentencepiece_albert_test.cpp`
  - Result: passed. Coverage loads the real local
    `albert-base-v1-tokenizer.json` and verifies Rust-derived ids, tokens,
    offsets, word ids, type ids, special-token masks, attention masks, pair
    TemplateProcessing, Metaspace decode, NFKD/StripAccents offset projection,
    ICU compatibility decomposition, ICU lowercase beyond the previous targeted
    tables, context-sensitive Greek final sigma lowercase, and `Precompiled`
    charsmap behavior for zero-width joiner/non-joiner splitting,
    no-break/BOM boundaries, chained zero-width deletion, record-separator
    deletion, compatibility expansion, and expansion-followed-by-removal/joiner
    projection.
- C++ RoBERTa processing runtime parity:
  - `tests/parity/roberta_processing_test.cpp`
  - Result: passed. Coverage loads the real local `roberta.json` and verifies
    Rust-derived single/pair ids, tokens, offsets, word ids, all-zero type ids,
    special-token masks, attention masks, ByteLevel decode, pair truncation
    accounting for the four-special-token RoBERTa shape, fixed padding over
    main/overflowing encodings, and pre-tokenized single/pair batch APIs with
    word-relative offsets.
- C++ post-processor Sequence runtime parity:
  - `tests/parity/post_processor_sequence_test.cpp`
  - Result: passed. Coverage builds a local `tokenizer.json` variant with
    `Sequence(ByteLevel, TemplateProcessing)` and verifies Rust-derived
    single/pair ids, tokens, offsets, word ids, type ids, special-token masks,
    attention masks, ByteLevel decode behavior, truncation + fixed padding, and
    pre-tokenized single/pair batch overloads with overflowing encodings.
- C++ generalized TemplateProcessing runtime parity:
  - `tests/parity/template_processing_test.cpp`
  - Result: passed. Coverage corresponds to
    `processors::template::tests::{template_processing,template_processing_overflowing}`
    for the accepted JSON surface: multi-id special tokens, non-default type
    ids, pair templates that reorder `A`/`B`, `add_special_tokens=false`
    sequence ordering/type-id behavior, multi-id special-token truncation
    accounting, and nested pair-overflow cross-products.
- C++ Llama encode runtime parity:
  - `tests/parity/llama_encode_test.cpp`
  - Result: passed. Coverage loads the real local `llama-3-tokenizer.json` and
    verifies Rust-derived single/pair ids, tokens, offsets, word ids, type ids,
    special-token masks, attention masks, three-digit split behavior,
    contractions, newline/multi-space splitting, Latin/CJK/fullwidth digit
    classification, default-ICU `AሴB Ⅻ3` category classification, and
    ByteLevel decode behavior. It now also injects truncation and fixed padding
    into a temporary real Llama JSON copy to verify single-sequence and pair
    overflowing, `<|begin_of_text|>` template insertion on main and overflow,
    the `<|finetune_right_pad_id|>` pad token, ordered `encode_batch` /
    `encode_batch_pairs` output, and char-offset batch/pair-batch output over
    the same truncation + padding path. It now also covers pre-tokenized
    single and pair batch APIs, preserving word-relative offsets and
    caller-supplied word ids through overflowing encodings.
- C++ Split pre-tokenizer runtime parity:
  - `tests/parity/split_pre_tokenizer_test.cpp`
  - Result: passed. Coverage corresponds to
    `pre_tokenizers::split::tests::{basic,regex_string,invert,serialization}`
    for the accepted runtime surface: String patterns, ICU Regex patterns,
    invert, delimiter behaviors, direct Split JSON, and WordLevel encode
    offsets. The default ICU build also checks a supplementary-plane
    `\p{P}+` regex match. This now also covers upstream Digits
    contiguous/individual modes and `Sequence(Whitespace, Digits)`.
- C++ documentation pipeline parity:
  - `tests/parity/documentation_pipeline_test.cpp`
  - Result: passed. Coverage corresponds to the inference-only portions of
    `documentation.rs::{quicktour,pipeline}` over real `tokenizer-wiki.json`.
- C++ documentation load parity:
  - `tests/parity/documentation_load_test.cpp`
  - Result: passed. Coverage corresponds to
    `documentation.rs::load_tokenizer` over real `roberta.json`.
- C++ Replace normalizer runtime parity:
  - `tests/parity/replace_normalizer_test.cpp`
  - Result: passed. Coverage corresponds to
    `normalizers::replace::tests::{test_replace,test_replace_regex,serialization}`
    for the accepted runtime surface: String patterns, ICU Regex patterns,
    direct Replace JSON, Unigram/SentencePiece encode offsets, and
    supplementary-plane `\p{P}+` replacement in the default ICU build.
- C++ stream decode runtime parity:
  - `tests/parity/stream_decode_test.cpp`
  - Result: passed. Coverage loads the real local `llama-3-tokenizer.json`,
    applies runtime ByteLevel normalizer and Split pre-tokenizer mutation,
    adds the upstream `嗎` and `д` tokens at runtime, verifies Rust-derived
    encode ids, tokens, offsets, word ids, type ids, masks, and decoded text,
    verifies `DecodeStream::step` buffering for partial Korean UTF-8 token
    fragments, checks `DecodeStream::has_pending()` across ByteLevel and
    ByteFallback partial-byte states, pins orphan ByteFallback replacement
    before the next regular token, and covers
    `documentation.rs::streaming_tokenizer` over real `roberta.json`, real
    `albert-base-v1-tokenizer.json`, and a local ByteFallback partial-byte
    fixture. R4 hardening now also covers stream behavior for
    `Sequence(ByteFallback,Fuse)`, `Sequence(Metaspace,Fuse)`, and
    `Sequence(ByteLevel,Fuse)`, including the no-finalize boundary where
    callers discard a stream to reset pending ByteFallback bytes.
- C++ decoder Sequence runtime parity:
  - `tests/parity/decoder_sequence_test.cpp`
  - Result: passed. Coverage corresponds to upstream decoder chain semantics
    for `Sequence(ByteFallback,Fuse)`, `Sequence(CTC,Metaspace)`,
    `Sequence(CTC,Fuse)` cleanup/custom-config behavior,
    `Sequence(Strip,Replace,Fuse)`, ICU Regex `Replace` inside a decoder
    chain, and direct `BPEDecoder`.
- C++ truncation runtime parity:
  - `tests/parity/truncation_test.cpp`
  - Result: passed. Coverage corresponds to
    `tokenizer::encoding::{truncate_overflow_with_stride,truncate_left}`,
    `tokenizer::tests::{right_truncation_early_exit_matches_full_encode,left_truncation_keeps_tail_tokens}`,
    `tokenizer::tests::{pair_right_truncation_longest_first,pair_only_second_does_not_truncate_first,pair_only_first_does_not_truncate_second}`,
    `utils::truncation::{truncate_encodings_longest_first,truncate_encodings_empty}`,
    and `utils::truncation::test_deserialize_defaults`.
- C++ padding runtime parity:
  - `tests/parity/padding_test.cpp`
  - Result: passed. Coverage corresponds to
    `tokenizer::encoding::padding`, `utils::padding::pad_to_multiple`, and the
    padding path from `documentation.rs::quicktour`, including native
    padding across public single-sequence, pair, direct pre-tokenized pair,
    pre-tokenized single-sequence batch, and pre-tokenized pair batch outputs.
- C++ tokenizer-level batch API smoke:
  - `tests/parity/tokenizer_api_smoke.cpp`
  - Result: passed. Coverage includes ordered `decode_batch` over regular and
    special-token-skipping decode paths, explicit byte-vs-char offset checks
    for Unicode text, char-offset single/pair/pre-tokenized APIs, char-offset
    batch APIs, and empty batch handling. The real Llama JSON smoke now also
    covers char-offset batch and pair-batch APIs with truncation, overflowing,
    template insertion, and fixed padding.
- C++ tokenizer-level batch execution boundary:
  - `docs/specs/batch-runtime-boundary.md`
  - Result: passed through the existing batch padding and API smoke tests.
    Batch encode/decode outputs remain index-stable, and `BatchLongest`
    padding still runs after individual encodings are produced.
- Upstream serialization reference:
  - `cargo test --test serialization`
  - Result: passed, `11/11` tests.
- Upstream added-token reference:
  - `cargo test --test added_tokens`
  - Result: passed, `5/5` tests. The environment still prints
    `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream ByteLevel/BPE reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test added_tokens lstrip_tokens -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test added_tokens rstrip_tokens -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_basic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_unicode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_double_sequence -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_pre_tokenized_sequence -- --exact`
  - Result: passed, `1/1` selected test for each command. The environment still
    prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream BERT added-token offset reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets split_on_added_tokens_bert -- --exact`
  - Result: passed, `1/1` selected test. A transient Rust field probe confirmed
    the exact ids, tokens, offsets, word ids, type ids, all-zero special-token
    mask, and attention mask now recorded in
    `docs/specs/split-on-added-tokens-bert.md`. The environment still prints
    `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream BertProcessing reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::bert::tests::bert_processing -- --exact`
  - Result: passed, `1/1` selected unit test. The environment still prints
    `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream WordPiece decoder reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test decoders::wordpiece::tests::wordpiece_decoder -- --exact`
  - Result: passed, `1/1` selected unit test. The environment still prints
    `/root/code/emsdk/emsdk: 39: exec: python: not found`, but Cargo exits
    successfully.
- Upstream decoder Sequence reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test decoders::sequence::tests::sequence_basic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test decoders::byte_fallback::tests::decode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test decoders::fuse::tests::decode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test decoders::strip::tests::decode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test normalizers::replace::tests::test_replace_decode -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream TemplateProcessing reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing_overflowing -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing_serde -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream BertNormalizer/BertPreTokenizer reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test normalizers::strip::tests::test_strip_accents -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::normalizer::tests::range_conversion -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::bert::tests::basic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::bert::tests::chinese_chars -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream BPE config reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::bpe::model::tests::test_unk_not_fused -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::bpe::model::tests::test_unk_get_fused -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::bpe::model::tests::test_bpe_with_continuing_subword_prefix -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::bpe::model::tests::test_bpe_byte_fallback -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::bpe::model::tests::test_ignore_merges -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream Unigram reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test test_unigram_from_file -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::unigram::model::tests::test_encode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test models::unigram::model::tests::test_unigram_bytefallback -- --exact`
  - Result: passed, `1/1` selected test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream SentencePiece/ALBERT component spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::metaspace::tests::basic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::whitespace::tests::whitespace_split -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test normalizers::precompiled::tests::expansion_followed_by_removal -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing_serde -- --exact`
  - Result: passed, `1/1` selected test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream RoBERTa/Template post-processor spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::roberta::tests::serde -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::roberta::tests::roberta_processing -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::template::tests::template_processing -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test processors::sequence::tests::process_chain -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream Llama pre-tokenizer spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::split::tests::basic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::byte_level::tests::pre_tokenization_no_regex -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream Split pre-tokenizer reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test pre_tokenizers::split::tests -- --nocapture`
  - Result: passed, `4/4` selected unit tests: `basic`,
    `regex_string`, `invert`, and `serialization`.
- Upstream Replace normalizer reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test normalizers::replace::tests -- --nocapture`
  - Result: passed, `4/4` selected unit tests: `test_replace`,
    `test_replace_regex`, `serialization`, and `test_replace_decode`.
- Upstream stream decode reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test stream test_decode_stream_step_no_panic -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test stream test_decoding_with_added_bpe -- --exact`
  - Result: passed, `1/1` selected test for each command.
- Upstream documentation streaming reference:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test documentation streaming_tokenizer -- --exact`
  - Result: passed, `1/1` selected test.
- Upstream truncation reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::encoding::tests::truncate_overflow_with_stride -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::encoding::tests::truncate_left -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::tests::right_truncation_early_exit_matches_full_encode -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::tests::left_truncation_keeps_tail_tokens -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::tests::pair_right_truncation_longest_first -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::tests::pair_only_second_does_not_truncate_first -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::tests::pair_only_first_does_not_truncate_second -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test utils::truncation::tests::truncate_encodings_longest_first -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test utils::truncation::tests::truncate_encodings_empty -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test utils::truncation::tests::test_deserialize_defaults -- --exact`
  - Result: passed, `1/1` selected unit test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.
- Upstream padding reference spot checks:
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test tokenizer::encoding::tests::padding -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test utils::padding::tests::pad_to_multiple -- --exact`
  - `CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test documentation quicktour -- --exact`
  - Result: passed, `1/1` selected test for each command. The environment
    still prints `/root/code/emsdk/emsdk: 39: exec: python: not found`, but
    Cargo exits successfully.

## Open Blockers

- Full Unicode casefold/category edge cases and broad regex behavior still need
  more component-by-component parity. Llama letter/number classification,
  BertNormalizer cleanup/lowercase/strip-accents, BertPreTokenizer
  punctuation, generic serialized Split regex matching, serialized Replace
  regex normalizers, and SentencePiece/ALBERT NFD/NFKD/lowercase now use the
  vendored ICU backend.
- Broader byte-level BPE coverage outside the accepted pair/pre-tokenized and
  BPE config boundaries remains a follow-up, including pre-tokenized trimming
  behavior and broader ByteLevel fixtures.
- Broader BERT coverage remains a follow-up beyond
  `offsets.rs::split_on_added_tokens_bert`, especially more normalization and
  mixed component edge cases.
- Standalone public `models::BPE`/fluent builder APIs, explicit cache
  resize/clear APIs, and exact stochastic dropout controls are deferred by
  `ADR-0004` unless a concrete downstream integration needs them. Unigram
  sampling, arbitrary
  multi-special post-processor composition, broad SentencePiece/Precompiled
  charsmap validation beyond the accepted ALBERT fixtures, and generalized
  runtime component mutation beyond the accepted stream slice still need
  component-by-component translation or explicit exclusion notes.
- Full upstream `cargo test` still includes intentionally excluded training and
  HTTP/from-pretrained surfaces.

## QA Checklist For R1 Dispatch Review

Scope under review:
`serialization.rs::{normalizers,processors,pretoks,decoders,models,tokenizer}` wrapper
dispatch. This review should verify native C++ JSON loading and error behavior
only; no Rust FFI, runtime shell-out, training, HTTP, benches, wrappers, or
patches to `third_party/tokenizers` are allowed.

Acceptance checks:

- Confirm C++ fixtures or inline test JSON are derived from the upstream
  serialized forms in `third_party/tokenizers/tokenizers/tests/serialization.rs`.
- Confirm wrapper dispatch is tested through the actual tokenizer/component load
  path, not only by constructing C++ concrete classes directly.
- Confirm positive coverage for `NFC`, default `BertNormalizer`,
  `BertPreTokenizer`, `CharDelimiterSplit`, `Whitespace`, `Split` with
  `String` and `Regex` patterns, exact upstream `BertProcessing`, `ByteLevel`
  decoder, default `BPE` model, and a full tokenizer JSON with `WordPiece` plus
  `NFC`.
- Confirm negative coverage for wrong concrete type, unsupported wrapper
  `type`, and malformed known fields.
- Confirm error messages identify the operation or component slot and offending
  `type` or field. Byte-for-byte Rust error strings are not required.
- Confirm implementation does not silently turn unsupported or malformed
  components into absent components or no-ops.
- Confirm `serialization.rs::processors` is covered by the exact upstream
  `BertProcessing` serde shape through the tokenizer load path.
- Confirm no C++ test requires byte-identical JSON serialization output; ADR-0002
  only requires load and inference parity right now.

Review commands:

- Upstream reference spot checks:
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization normalizers -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization pretoks -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization decoders -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization models -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization tokenizer -- --exact`
- C++ build and parity tests:
  - `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build -DTOKENIZERS_CPP_BUILD_TESTS=ON`
  - `cmake --build projects/tokenizers.cpp/build`
  - `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`
- Fixture freshness, if the implementation updates generated fixtures:
  - `uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check`

Failure modes to flag:

- Dispatch succeeds for the wrong concrete type, such as accepting `NFC` JSON as
  `NFKC` or `Whitespace` JSON as `BertPreTokenizer`.
- Unsupported `type` values load without failure.
- Missing or malformed required fields are ignored.
- A component is parsed but its configured fields are lost before runtime use.
- Regex-vs-string `Split` pattern forms collapse into the same representation.
- Tests pass only because they inspect JSON strings or constructors instead of
  loading through the production parser.
- New code reaches into Rust, shells out at runtime, adds training or HTTP
  surfaces, or modifies `third_party/tokenizers`.

## QA Checklist For R1 Model JSON Load Review

Scope under review:
`serialization.rs::{bpe_serde,wordpiece_serde,wordlevel_serde,test_deserialize_long_file}`
model JSON load coverage plus the dependency switch to a local copied
`nlohmann/json.hpp`. This review is docs, fixtures, native C++ loader, and CMake
only; no Rust FFI, runtime shell-out, training, HTTP/from-pretrained, benches,
wrappers, or patches to `third_party/tokenizers` are allowed.

Acceptance checks:

- Confirm `projects/tokenizers.cpp` carries its own copied
  `include/nlohmann/json.hpp` derived from
  `projects/magic-tts-ggml-cpp/vendor/nlohmann/json.hpp`.
- Confirm CMake uses the local header and has no configure-time network path for
  `nlohmann_json`: no `FetchContent_Declare(nlohmann_json)`, no
  `GIT_REPOSITORY https://github.com/nlohmann/json.git`, and no requirement for
  a system `nlohmann_json` package.
- Confirm a clean configure passes with dependency fetching disabled.
- Confirm BPE, WordPiece, and WordLevel JSON fixtures are derived from upstream
  `serialization.rs::{bpe_serde,wordpiece_serde,wordlevel_serde}` and are loaded
  through the production tokenizer/model parser.
- Confirm the large-file test loads the same class of full tokenizer JSON used
  by upstream `test_deserialize_long_file`, not a minimized substitute.
- Confirm tests assert meaningful parsed model state and selected stable
  vocabulary/id lookups for BPE, WordPiece, and WordLevel instead of only
  checking that JSON parses as a generic object. Do not use encode/decode output
  as acceptance for this load-only slice.
- Confirm unsupported fields or model features are named as explicit gaps and
  either rejected with clear errors or covered by tests. Silent field loss is not
  acceptable.
- Confirm JSON serialization output parity is not required for this slice unless
  the implementation adds a serializer in the same change.

Review commands:

- Local header and no-network dependency audit:
  - `test -f projects/tokenizers.cpp/include/nlohmann/json.hpp`
  - `cmp -s projects/magic-tts-ggml-cpp/vendor/nlohmann/json.hpp projects/tokenizers.cpp/include/nlohmann/json.hpp`
  - `! rg -n "FetchContent_Declare\\(\\s*nlohmann_json|github.com/nlohmann/json|find_package\\(nlohmann_json" projects/tokenizers.cpp/CMakeLists.txt projects/tokenizers.cpp/cmake`
- Upstream reference spot checks:
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization normalizers -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization processors -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization pretoks -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization decoders -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization models -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization tokenizer -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization bpe_serde -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization wordpiece_serde -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization wordlevel_serde -- --exact`
  - `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test serialization test_deserialize_long_file -- --exact`
- C++ no-fetch configure, build, and parity tests:
  - `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
  - `cmake --build projects/tokenizers.cpp/build`
  - `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`
- Fixture freshness, if the implementation updates generated fixtures:
  - `uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check`

Failure modes to flag:

- Configure succeeds only because CMake downloads or discovers an external
  `nlohmann_json` package.
- A model JSON document loads after dropping unsupported known fields without an
  explicit error or documented gap.
- Tests instantiate model classes directly and bypass the production JSON load
  path.
- The large-file test uses a tiny hand-written JSON document instead of a
  realistic full tokenizer fixture.
- New BPE fields are accepted without being represented in model state, runtime
  tests, or review notes.
- New code reaches into Rust, shells out at runtime, adds training or HTTP
  surfaces, or modifies `third_party/tokenizers`.

## R2 SentencePiece Unicode Normalization Slice

Current state:

- The native C++ ALBERT/SentencePiece path now executes ordered simple
  normalizer sequences for `Replace`, ICU-backed `NFC`/`NFD`/`NFKC`/`NFKD`,
  upstream `Nmt`, serialized `Prepend`, Unicode-whitespace `Strip`,
  `StripAccents`, ICU-backed string-level `Lowercase`, and the covered
  serialized `Precompiled` charsmap path.
- The ALBERT runtime no longer uses the previous targeted decomposition table
  for `NFD`/`NFKD`; those normalizer steps now route through the vendored static
  ICU backend. The project default build links the vendored static ICU backend,
  and audit checks did not find system `libicu*.so` dependencies.
- `tests/parity/sentencepiece_albert_test.cpp` now covers non-ASCII ALBERT and
  minimal SentencePiece fixtures:
  - `Héllò hôw are ü?` -> `▁hello ▁how ▁are ▁u ?`
  - `ậ…` -> `▁a . . .`
  - Thai `ำน้ำ3ลำ` shape from the upstream strip bug path -> `▁ านา 3 ลา`
  - minimal ICU compatibility decomposition shape: `① ㍿` -> `1 株式会社`
  - minimal ICU lowercase shape: `Ա Բ` -> `ա բ`
  - context-sensitive lowercase shape: `ΟΣ` -> `ος`
  - `A\u200dB`, `A\u200cB`, `A\u200bB`, and `hello\u200dworld` zero-width
    joiner/non-joiner/space behavior from the serialized `Precompiled`
    charsmap.
  - `hello\u001eworld` record-separator deletion without a new word boundary.
  - `™\u001eg` expansion-followed-by-removal projection inspired by upstream
    `normalizers/precompiled.rs::expansion_followed_by_removal`.
  - `A\u000bB\u000cC` control separator mapping/deletion and
    `e\u0301\u200dg` StripAccents followed by joiner splitting, both generated
    from the upstream Rust ALBERT tokenizer.
  - R4 hardening fixtures generated from upstream Rust ALBERT output:
    `A\u00a0B`, `A\ufeffB`, `A\u200d\u200c\u200bB`, `ﬃ`, and `™\u200dg`,
    covering no-break/BOM separator boundaries, chained zero-width deletion,
    compatibility expansion, and expansion followed by joiner splitting.
- `tests/parity/replace_normalizer_test.cpp` now also covers the generic
  Unicode normalizer path with `Sequence(Strip, NFC)` over decomposed accented
  input, direct `NFKC` compatibility expansion, direct `Nmt`
  removal/space-mapping, and direct `Prepend` non-empty/empty-input behavior.
- The default vendored ICU build passes 31/31 tests, including the shared-ICU
  dependency audit. There is no no-ICU build path in the supported matrix.

Remaining gaps:

- Full Unicode database use is now wired into the accepted Llama
  letter/number classification, BertNormalizer cleanup/lowercase/strip-accents,
  BertPreTokenizer punctuation, SentencePiece
  `NFC`/`NFD`/`NFKC`/`NFKD`/`Nmt`/`Prepend`/lowercase, and serialized
  Split/Replace regex surfaces, including the accepted
  ByteLevel/BPE regex-composition fixtures. R4-H1 broadened `Precompiled`
  validation over the local ALBERT charsmap; additional charsmap validation is
  now downstream-driven if more real SentencePiece tokenizer JSON fixtures are
  added.
- ICU offset projection for `NFD`/`NFKD` is per-originating-codepoint in the
  tokenizer layer. `NFC`/`NFKC` now groups a base codepoint plus following
  nonspacing marks for common composition cases. Broader cross-base canonical
  reordering remains a follow-up if a fixture needs byte-exact upstream behavior
  there.
- Future full regex dependencies must not be introduced as default system
  dynamic dependencies.

## Next Actions

1. Keep BPE work behind the accepted tokenizer-centered public surface. Re-open
   `ADR-0004` only if a downstream C++/GGML integration needs standalone
   `models::BPE`, builder APIs, explicit cache controls, or exact stochastic
   dropout controls.
2. Treat R5 consumer readiness as complete for generic integration. Add a
   model-specific real-tokenizer consumer smoke only when a downstream GGML/C++
   project chooses concrete tokenizer JSON fixtures.
3. Keep any additional decoder composition follow-ups focused on real tokenizer
   JSON demand; the R4-H1 ByteFallback/Fuse, Metaspace/Fuse, ByteLevel/Fuse,
   and no-finalize stream boundaries are now pinned.
4. Add rare ICU offset projection fixtures only when a concrete tokenizer
   exposes byte-exact upstream behavior that the current grouping does not
   cover.
5. Add future performance work only behind private implementation boundaries
   unless a downstream C++/GGML integration re-opens the public surface.
