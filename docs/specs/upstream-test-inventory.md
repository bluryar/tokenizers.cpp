# Upstream Test Inventory

Source: `third_party/tokenizers/tokenizers/tests` at
`22d54d37621f2d9f35cf9420d6ed8658372a6c5d`.

| Upstream file/test | Classification | C++ target |
| --- | --- | --- |
| `added_tokens.rs::add_tokens` | port now | Added vocabulary and special-token id assignment |
| `added_tokens.rs::lstrip_tokens` | current byte-level slice | Added-token lstrip, byte-level BPE tokens, Unicode byte offsets |
| `added_tokens.rs::rstrip_tokens` | current byte-level slice | Added-token rstrip and byte-level prefix-space interaction |
| `added_tokens.rs::single_word_tokens` | port now | Added-token single-word boundary matching |
| `added_tokens.rs::overlapping_tokens` | port now | Added-token leftmost-longest trie precedence |
| `offsets.rs::byte_level_basic` | current byte-level slice | Byte-level offsets with and without trimming |
| `offsets.rs::byte_level_unicode` | current byte-level slice | Multi-byte Unicode byte-level offsets |
| `offsets.rs::byte_level_double_sequence` | current pair/pre-tokenized slice | Pair ids, tokens, offsets, word ids, type ids, and masks |
| `offsets.rs::byte_level_pre_tokenized_sequence` | current pair/pre-tokenized slice | Pre-tokenized byte-level ids, tokens, offsets, word ids, type ids, and masks |
| `offsets.rs::byte_level_pre_tokenized_sequence_with_trimming` | port | Pre-tokenized trimming behavior; upstream marks this test ignored |
| `offsets.rs::split_on_added_tokens_bert` | covered BERT added-token slice | Added special-token split ids, offsets, tokens, word ids, and masks |
| `models/wordpiece::tokenize` | covered runtime source | Greedy subword, unknown-token, and max-input-character behavior |
| `decoders/wordpiece::wordpiece_decoder` | covered runtime source | Decoder prefix handling, initial-prefix preservation, and cleanup field behavior |
| `decoders/sequence::sequence_basic` | covered runtime source | Decoder `Sequence` chain semantics for CTC followed by Metaspace |
| `decoders/byte_fallback::decode` | covered narrow runtime source | Valid and invalid `<0xNN>` byte-token runs through `Sequence(ByteFallback,Fuse)` |
| `decoders/fuse::decode` | covered runtime source | Token-list fusion in decoder chains |
| `decoders/strip::decode` | covered narrow runtime source | Single-codepoint left/right strip semantics in decoder chains |
| `decoders/bpe::BPEDecoder` | covered narrow runtime source | Word-end suffix replacement and final-token handling |
| `normalizers/replace::test_replace_decode` | covered narrow runtime source | Replace decoder String and ICU Regex patterns inside decoder chains |
| `models/wordlevel::{test_tokenize_unk,test_tokenize_missing_unk_token}` | covered runtime source | WordLevel unknown fallback and missing-unk failure |
| `models/bpe::{test_unk_not_fused,test_unk_get_fused}` | covered runtime source | BPE unknown fallback with and without `fuse_unk` |
| `models/bpe::test_bpe_with_continuing_subword_prefix` | covered runtime source | BPE continuing-subword prefix lookup and merge output derivation |
| `models/bpe::test_bpe_byte_fallback` | covered runtime source | BPE byte fallback and fallback-to-unk behavior |
| `models/bpe::test_bpe_byte_fallback_newline` | covered runtime source | Raw non-ByteLevel BPE newline byte fallback |
| `models/bpe::test_ignore_merges` | covered runtime source | BPE full-vocab hit when `ignore_merges=true` |
| `models/bpe::test_cache_is_per_bpe_instance` | covered narrow runtime source | C++ R6 BPE cache keeps a private per-tokenizer cache id; fixture pins independent tokenizer instance behavior and repeated-piece offset projection |
| `local::bpe_heap_stale_candidate` | local hardening | C++ R6 deterministic BPE merge heap preserves rank order and rejects stale adjacency candidates after neighboring merges |
| `models/bpe::{test_bpe_from_file,test_bpe_from_file_bad_merges,test_bpe_from_file_merge_token_oov}` | covered narrow runtime source | Public `Tokenizer::from_bpe_files` covers vocab/merges files, bad merge lines, and merge-token OOV; tokenizer JSON load also rejects OOV merges |
| `models/bpe::test_tokenize_with_and_without_dropout` | covered narrow runtime source | Deterministic dropout boundaries: `None`/`0.0` preserve merges and `1.0` skips all merges; stochastic middle probability is covered by shape invariants, not exact token sequences |
| `processors/bert::bert_processing` | covered runtime source | `[CLS]`/`[SEP]` insertion for single and pair encodings, plus real BERT JSON truncation/padding over raw and pre-tokenized batch APIs |
| `processors/roberta::{serde,roberta_processing}` | covered runtime source | RoBERTa JSON load, `<s> A </s></s> B </s>` pair processing, all-zero type ids, ByteLevel offset trimming, and real RoBERTa truncation/padding over raw and pre-tokenized overflowing encodings |
| `processors/sequence::process_chain` | covered runtime source | Post-processor Sequence composition for covered ByteLevel and TemplateProcessing children, including truncation/padding over raw and pre-tokenized overflowing encodings |
| `normalizers/bert::{clean_text,handle_chinese_chars,strip_accents,lowercase}` | covered runtime source | Default BERT normalization order, Unicode-backend control/whitespace handling, Chinese spacing, NFD accent stripping, and lowercase; default ICU build adds Rust-derived non-ASCII cleanup coverage |
| `normalizers/replace::{test_replace,test_replace_regex,serialization}` | covered runtime source | Replace String and ICU Regex patterns, JSON load, Unigram encode-side offsets, default-ICU supplementary-plane regex matching, and narrow ByteLevel/BPE composition |
| `normalizers/utils::Lowercase` | covered runtime source | ICU-backed string-level lowercase with edits in BERT and SentencePiece paths, including context-sensitive Greek final sigma; casefold remains a follow-up |
| `pre_tokenizers/bert::{basic,chinese_chars}` | covered runtime source | Whitespace splitting, Chinese spacing from BertNormalizer, and punctuation isolation through the shared Unicode backend; default ICU build adds supplementary-plane punctuation coverage |
| `pre_tokenizers/whitespace::{basic,whitespace_split}` | covered runtime source | Direct Whitespace grouping through ICU regex in the BERT documentation pipeline and WhitespaceSplit offsets in the ALBERT SentencePiece path |
| `pre_tokenizers/digits::{numbers,individual_digits}` | covered runtime source | Direct Digits pre-tokenizer contiguous and individual modes, plus `Sequence(Whitespace, Digits)` order in the documentation pipeline |
| `pre_tokenizers/metaspace::basic` | covered runtime source | Metaspace replacement, prefix marker, split, and offsets for ALBERT |
| `pre_tokenizers/split::{basic,regex_string,invert,serialization}` | covered runtime source | Direct and Sequence Split JSON, String and ICU Regex patterns, invert, all five delimiter behaviors, byte offsets, and narrow ByteLevel/BPE composition |
| `pre_tokenizers/byte_level::pre_tokenization_no_regex` | covered runtime source | ByteLevel(use_regex=false) after upstream Split, preserving each split piece |
| `processors/template::{template_processing,template_processing_overflowing}` | covered runtime source | Direct serialized TemplateProcessing sequence/special-token pieces, including ALBERT/BERT shape, multi-id specials, non-default type ids, sequence reordering, special-token accounting, and nested pair-overflow cross-products |
| `decoders/metaspace::Metaspace` | covered runtime source | Metaspace marker-to-space decode for ALBERT |
| `normalizers/unicode::{NFC,NFD,NFKC,NFKD}` | covered runtime source | Vendored ICU normalization for tokenizer normalizer sequences, including `NFC` composition, `NFKC` compatibility expansion fixtures beyond the previous targeted table, and real `bert-wiki.json` `Sequence(NFD,Lowercase,StripAccents)` before WordPiece |
| `normalizers/unicode::Nmt` | covered narrow runtime source | Exact upstream NMT control removal and whitespace mapping with byte-offset projection |
| `normalizers/prepend::test_prepend` | covered narrow runtime source | Non-empty `Prepend` insertion, empty-input no-op, and first-codepoint span projection |
| `normalizers/strip::Strip` | covered narrow runtime source | Unicode-whitespace leading/trailing trim with byte-offset projection in simple normalizer sequences |
| `normalizers/strip::{test_strip_accents,test_vietnamese_bug,test_thai_bug,test_strip_accents_multiple}` | covered narrow runtime source | Combining-mark removal subset; C++ ALBERT fixtures cover Latin accents, Vietnamese `ậ`, ellipsis, and offset projection |
| `normalizers/precompiled::expansion_followed_by_removal` | covered runtime source | Serialized SentencePiece charsmap load plus double-array transform; C++ ALBERT fixtures cover zero-width joiner/non-joiner/space splitting, chained zero-width deletion, no-break/BOM separator boundaries, control separator mapping/deletion, record-separator deletion, StripAccents followed by joiner splitting, compatibility expansion, and expansion-followed-by-removal/joiner offset projection |
| `serialization.rs::bpe_serde` | covered load source | Real GPT-2 BPE vocab/merges JSON load and stable id/token lookups |
| `serialization.rs::wordpiece_serde` | covered load source | Real BERT WordPiece vocab JSON load and stable id/token lookups |
| `serialization.rs::wordlevel_serde` | covered load source | Real GPT-2 WordLevel vocab JSON load and stable id/token lookups |
| `serialization.rs::normalizers` | covered dispatch source | `NFC`, default `BertNormalizer`, and wrong-slot rejection through tokenizer JSON load |
| `serialization.rs::processors` | covered dispatch source | Exact upstream `BertProcessing` serde shape and wrapper dispatch through tokenizer JSON load |
| `serialization.rs::pretoks` | covered dispatch source | `BertPreTokenizer`, `CharDelimiterSplit`, `Whitespace`, and String/Regex `Split` wrapper dispatch |
| `serialization.rs::decoders` | covered dispatch source | Default `ByteLevel` decoder wrapper dispatch and decoder `Sequence` rejection/acceptance cases |
| `serialization.rs::models` | covered dispatch source | BPE model wrapper dispatch plus wrong-model rejection through tokenizer JSON load |
| `serialization.rs::tokenizer` | covered dispatch source | Full WordPiece tokenizer JSON with `NFC` normalizer wrapper deserialization |
| `serialization.rs::bpe_with_dropout_serde` | covered runtime source | BPE dropout field JSON load and validation, with deterministic encode boundaries covered |
| `serialization.rs::test_deserialize_long_file` | covered load source | Real ALBERT tokenizer JSON deserialize smoke |
| `stream.rs::test_decoding_with_added_bpe` | covered narrow runtime source | Llama ByteLevel decode plus runtime ByteLevel normalizer, Split pre-tokenizer, and `add_tokens` mutation for the upstream added-BPE encode/decode fixture |
| `stream.rs::test_decode_stream_step_no_panic` | covered runtime source | Stateful ByteLevel decode stream buffers partial UTF-8 tokens before emission |
| `tokenizer/mod::{encode_batch,decode_batch}` | covered runtime source | Ordered tokenizer-level batch execution; per-item work can run independently, with batch padding applied after all encodings are produced, including real Llama pre-tokenized batch variants |
| `tokenizer/mod::{encode_char_offsets,encode_batch_char_offsets}` | covered narrow runtime source | Explicit char-offset APIs project UTF-8 byte offsets to scalar indexes across single, pair, pre-tokenized, batch, and real BERT/Llama truncation/padding overloads |
| `tokenizer/encoding::{truncate_overflow_with_stride,truncate_left}` | covered narrow runtime source | Single-sequence right/left truncation and stride overflowing |
| `tokenizer/mod::{right_truncation_early_exit_matches_full_encode,left_truncation_keeps_tail_tokens}` | covered narrow runtime source | Single-sequence tokenizer-level truncation direction |
| `utils/truncation::test_deserialize_defaults` | covered runtime source | Old truncation JSON defaults missing `direction` to Right |
| `utils/truncation::truncate_encodings_longest_first` | covered narrow runtime source | Pair truncation length allocation and overflow merge behavior |
| `utils/truncation::truncate_encodings_empty` | covered narrow runtime source | Empty truncation across single/pair inputs |
| `tokenizer/encoding::padding` | covered runtime source | Left/right vector padding over ids, type ids, tokens, offsets, masks, and word ids |
| `utils/padding::pad_to_multiple` | covered runtime source | Fixed and BatchLongest padding length rounded to `pad_to_multiple_of`, including native single-sequence, pair, direct pre-tokenized pair, and pre-tokenized batch outputs |
| `unigram.rs::test_unigram_from_file` | covered runtime source | Unigram model load and trie-backed deterministic best-path inference |
| `models/unigram::{test_encode,test_unigram_bytefallback}` | covered runtime source | Unigram optimized best path, fused unknowns, byte fallback, repeated-piece cache offset preservation, and tokenizer-instance cache separation |
| `unigram.rs::test_sample` | reference-only | Uses sampling; keep as diagnostic unless deterministic parity is required |
| `unigram.rs::test_train_unigram_from_file` | skip-training | Trainer scope excluded |
| `training.rs::bpe_values_after_training` | skip-training | Trainer scope excluded |
| `training.rs::bpe_continuing_subword_prefix_error` | skip-training | Trainer scope excluded |
| `from_pretrained.rs::test_from_pretrained` | skip-non-core | HTTP/from-pretrained remote loading excluded |
| `from_pretrained.rs::test_from_pretrained_revision` | skip-non-core | HTTP/from-pretrained remote loading excluded |
| `from_pretrained.rs::test_from_pretrained_invalid_model` | skip-non-core | HTTP/from-pretrained remote loading excluded |
| `from_pretrained.rs::test_from_pretrained_invalid_revision` | skip-non-core | HTTP/from-pretrained remote loading excluded |
| `documentation.rs::train_tokenizer` | skip-training | Documentation trainer example excluded |
| `documentation.rs::quicktour_slow_train` | skip-training | Documentation trainer example excluded; upstream marks this test ignored |
| `documentation.rs::train_pipeline_bert` | skip-training | Documentation trainer example excluded; upstream marks this test ignored |
| `documentation.rs::load_tokenizer` | covered runtime source | Real `roberta.json` load, encode ids/tokens, and decode roundtrip |
| `documentation.rs::streaming_tokenizer` | covered runtime source | RoBERTa ByteLevel stream segments, ALBERT Metaspace first-token behavior, and ByteFallback partial-byte `None` behavior |
| `documentation.rs::quicktour` | covered runtime source | Real `tokenizer-wiki.json` initial encode, TemplateProcessing single/pair, ordered batch/batch-pair execution, BatchLongest padding tokens, and attention masks |
| `documentation.rs::pipeline` | covered runtime source | Real `tokenizer-wiki.json` with `Sequence(NFD,StripAccents)`, `Sequence(Whitespace,Digits)`, TemplateProcessing, encode fields, and decode |
| `documentation.rs::pipeline_bert` | covered runtime source | Real `bert-wiki.json` encode fields, direct Whitespace pre-tokenization, no-decoder WordPiece token joining, and runtime WordPiece decoder cleanup |
| `common/mod.rs` | reference-only | Upstream Rust fixture helper, not a C++ test by itself |

Milestone order:

1. `serialization.rs::{bpe_serde,wordpiece_serde,wordlevel_serde,test_deserialize_long_file}`
   to cover real model JSON loading. See `model-json-load.md`; this is load
   integrity and dispatch coverage, not tokenization behavior parity.
2. `added_tokens.rs::{add_tokens,single_word_tokens,overlapping_tokens}` to
   pin added-token id assignment, matching boundaries, and leftmost-longest
   precedence.
3. `added_tokens.rs::{lstrip_tokens,rstrip_tokens}` plus
   `offsets.rs::{byte_level_basic,byte_level_unicode}` as the first byte-level
   BPE runtime slice. See `byte-level-bpe-runtime-boundary.md` for exact
   accepted component behavior and gaps.
4. `offsets.rs::{byte_level_double_sequence,byte_level_pre_tokenized_sequence}`
   coverage now pins pair and pre-tokenized ids, tokens, offsets, word ids,
   type ids, and masks exactly. See
   `byte-level-pair-pretokenized-offsets.md`.
5. `offsets.rs::split_on_added_tokens_bert` is now covered as the first
   Bert-style path, broadening offset, token, word-id, and all-zero
   special-mask coverage into `BertNormalizer`, `BertPreTokenizer`, and
   WordPiece runtime. See `split-on-added-tokens-bert.md`.
6. WordPiece runtime breadth, WordPiece decoder cleanup, BertProcessing
   insertion, `BertNormalizer` cleanup/lowercase/strip-accents through the
   shared Unicode backend, and real `bert-wiki.json` truncation/padding over
   raw/pre-tokenized batch plus char-offset APIs are now covered in
   `wordpiece-bert-processing-breadth.md`.
7. WordLevel unknown fallback is now covered in
   `wordlevel-runtime-boundary.md`.
8. BPE runtime config breadth is now covered in
   `bpe-config-runtime-boundary.md`.
9. Unigram deterministic inference is now covered in
   `unigram-runtime-boundary.md`.
10. SentencePiece/ALBERT JSON runtime smoke, ICU-backed `NFD`/`NFKD`, and the
   covered `Precompiled` charsmap path, including the upstream-inspired
   expansion-followed-by-removal shape and Rust-derived ALBERT zero-width/control
   separator cases, are now covered in
   `sentencepiece-albert-runtime-boundary.md`.
11. RoBERTa post-processing and direct serialized TemplateProcessing pieces
   are now covered in `roberta-processing-runtime-boundary.md`.
12. Post-processor `Sequence(ByteLevel, TemplateProcessing)` runtime
   composition, including GPT-style pre-tokenized single/pair batch overloads
   under truncation, overflowing, and fixed padding, is now covered in
   `post-processor-sequence-runtime-boundary.md`.
13. Llama pre-tokenizer `Sequence(Split, ByteLevel(use_regex=false))`, real
   encode smoke, contractions, newline splitting, and Unicode letter/number
   category fixtures are now covered in
   `llama-split-bytelevel-runtime-boundary.md`. Default vendored ICU builds
   include a Rust-derived `AሴB Ⅻ3` fixture. The same real Llama JSON path now
   has temporary truncation + fixed-padding coverage for single and pair
   overflowing encodings plus ordered `encode_batch`, `encode_batch_pairs`,
   pre-tokenized batch/pair-batch, `encode_batch_char_offsets`, and
   `encode_batch_pairs_char_offsets`.
14. `stream.rs::*` decode behavior and the accepted encode-side added-BPE
   mutator path are now covered in `stream-decode-runtime-boundary.md`.
15. Real RoBERTa, GPT-style, and Llama tokenizer JSON smoke paths now compose
   BPE, ByteLevel post-processing, special-token insertion, truncation,
   overflowing, fixed padding, ordered batch/pair-batch output, char-offset
   output, and pre-tokenized output where covered through
   `roberta-processing-runtime-boundary.md`,
   `post-processor-sequence-runtime-boundary.md`,
   `llama-split-bytelevel-runtime-boundary.md`,
   `truncation-runtime-boundary.md`, and `padding-runtime-boundary.md`.

## Added Tokens Audit

Source audited:
`third_party/tokenizers/tokenizers/tests/added_tokens.rs`.

Classification:

- Port now:
  `add_tokens`, `single_word_tokens`, and `overlapping_tokens`. These are core
  added-vocabulary runtime behavior and do not primarily assert byte offsets.
- Current byte-level slice:
  `lstrip_tokens` and `rstrip_tokens`. These depend on byte-level BPE
  tokenization, whitespace capture, prefix-space behavior, and, for
  `lstrip_tokens`, exact Unicode byte offsets. The accepted native boundary is
  now defined in `byte-level-bpe-runtime-boundary.md`.
- Skip-training/non-core/reference-only:
  none for this file. All five tests are inference/runtime tokenizer behavior.

Current C++ coverage:

- `tests/parity/tokenizer_api_smoke.cpp` loads one JSON `added_tokens` entry for
  `[CLS]`, checks `token_to_id`, special-token mask, and decode skipping.
- `tests/parity/generated_fixture_test.cpp` exercises a generated WordLevel +
  WhitespaceSplit fixture whose tokenizer JSON includes the same simple
  `[CLS]` added token.
- `tests/parity/added_tokens_test.cpp` covers upstream-style runtime id
  assignment from JSON records, decode skipping, `single_word` matching, and
  leftmost-longest overlap precedence on the current WordLevel path. It also
  includes partial lstrip/rstrip whitespace-span checks, but not full upstream
  byte-level BPE parity.
- `tests/parity/byte_level_bpe_test.cpp` now loads local GPT-2 vocab/merges
  through tokenizer JSON and executes the production ByteLevel pre-tokenizer,
  deterministic BPE model, ByteLevel post-processor, and ByteLevel decoder path
  for `added_tokens.rs::{lstrip_tokens,rstrip_tokens}` and
  `offsets.rs::{byte_level_basic,byte_level_unicode}`. It also contains the
  accepted generic ICU regex composition smoke for `Replace` and `Split`
  before ByteLevel/GPT-2 BPE.

Concrete gaps:

- No public `AddedToken`, `add_tokens`, or `add_special_tokens` mutator API; this
  remains intentional for the current inference API boundary.
- `normalized=true` records are stored but do not yet match through a real
  normalizer output path.
- The ByteLevel/BPE runtime path is intentionally narrow: it covers direct
  ByteLevel slots, the accepted `Sequence(Split, ByteLevel)` regex-composition
  smoke, and deterministic GPT-2 BPE for this slice, not every pre-tokenizer
  sequence shape or every BPE configuration.
- Full Unicode letter/number category behavior in the GPT-2 regex splitter is
  not implemented yet; the accepted fixtures only require ASCII words/numbers
  and non-ASCII symbol byte mapping.
- Pair offsets and pre-tokenized byte-level inputs now have an accepted native
  boundary in `byte-level-pair-pretokenized-offsets.md`. BPE unknown fallback,
  byte fallback, prefix/suffix behavior, and `ignore_merges` have a separate
  accepted native boundary in `bpe-config-runtime-boundary.md`, including
  deterministic dropout boundaries. Pre-tokenized trimming, broad
  `use_regex=false` fixtures, stochastic middle-probability dropout fixtures,
  and broad decoder composition remain follow-ups.

The accepted boundary for the current ByteLevel slice is defined in
`byte-level-bpe-runtime-boundary.md`. It intentionally covers only the
ByteLevel + deterministic GPT-2 BPE behavior needed by
`added_tokens.rs::{lstrip_tokens,rstrip_tokens}` and
`offsets.rs::{byte_level_basic,byte_level_unicode}`.

## QA Audit: Strip Tokens And Byte-Level Offsets

Source audited:

- `third_party/tokenizers/tokenizers/tests/added_tokens.rs::{lstrip_tokens,rstrip_tokens}`
- `third_party/tokenizers/tokenizers/tests/offsets.rs::{byte_level_basic,byte_level_unicode}`
- Shared upstream fixtures in
  `third_party/tokenizers/tokenizers/tests/common/mod.rs`
- Current C++ tests in `tests/parity`

Upstream fixture requirements:

- `get_byte_level(add_prefix_space, trim_offsets)` uses GPT-2 BPE files
  `data/gpt2-vocab.json` and `data/gpt2-merges.txt`.
- The tokenizer has a `ByteLevel` pre-tokenizer with the requested
  `add_prefix_space`, a default `ByteLevel` decoder, and a `ByteLevel`
  post-processor with the requested `trim_offsets`.
- Because the C++ public API is inference-only, equivalent C++ parity should use
  tokenizer JSON or generated fixtures representing the same runtime state,
  not upstream Rust mutation APIs.

Exact acceptance:

- `added_tokens.rs::lstrip_tokens`:
  - Fixture: `get_byte_level(true, false)` plus special added token
    `<mask>` with `lstrip=true`.
  - Input: `I saw a <mask> 😺`.
  - Accept only exact tokens
    `["ĠI", "Ġsaw", "Ġa", " <mask>", "ĠðŁĺ", "º"]`.
  - Accept only exact byte offsets
    `[(0, 1), (1, 5), (5, 7), (7, 14), (14, 19), (15, 19)]`.
- `added_tokens.rs::rstrip_tokens`:
  - Fixture A: `get_byte_level(false, false)` plus special added token
    `<mask>` with `rstrip=true`.
  - Input: `I saw a <mask> 😺`.
  - Accept only exact tokens
    `["I", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ðŁĺ", "º"]`.
  - Fixture B: `get_byte_level(true, false)` with the same added token and
    input.
  - Accept only exact tokens
    `["ĠI", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ĠðŁĺ", "º"]`.
  - Upstream does not assert offsets for this test; if C++ adds offset
    assertions, generate them from the Rust reference and store them as fixture
    data.
- `offsets.rs::byte_level_basic`:
  - Fixture A: `get_byte_level(true, false)`.
  - Input: `Hello there, how are you?`.
  - Accept offsets whose input slices are exactly
    `["Hello", " there", ",", " how", " are", " you", "?"]`, i.e.
    `[(0, 5), (5, 11), (11, 12), (12, 16), (16, 20), (20, 24), (24, 25)]`.
  - Fixture B: `get_byte_level(true, true)`.
  - Accept trimmed offsets whose input slices are exactly
    `["Hello", "there", ",", "how", "are", "you", "?"]`, i.e.
    `[(0, 5), (6, 11), (11, 12), (13, 16), (17, 20), (21, 24), (24, 25)]`.
- `offsets.rs::byte_level_unicode`:
  - Fixture: `get_byte_level(true, false)`.
  - Input: `i⭢j`.
  - Accept that output offsets at indexes `1`, `2`, and `3` all slice the same
    original multibyte scalar `⭢`; in byte offsets this is `(1, 4)` for all
    three entries.

Current C++ coverage comparison:

- `tests/parity/added_tokens_test.cpp::test_lstrip_and_rstrip_matching`
  verifies leading/trailing whitespace span capture for `<mask>` on the current
  WordLevel + `WhitespaceSplit` path using `hello <mask> world`.
- `tests/parity/byte_level_bpe_test.cpp` is the exact C++ target for this
  audit. It covers the GPT-2 ByteLevel/BPE tokenizer JSON shape, exact
  `lstrip_tokens` tokens and offsets, exact `rstrip_tokens` tokens for both
  prefix-space modes, and exact `byte_level_basic`/`byte_level_unicode`
  offsets.
- `tests/parity/json_wrapper_dispatch_test.cpp` accepts `ByteLevel` wrapper
  JSON for pre-tokenizer, post-processor, and decoder slots as dispatch-only
  coverage. It does not execute ByteLevel behavior.
- `tests/parity/generated_fixture_test.cpp` and
  `tests/parity/tokenizer_api_smoke.cpp` assert simple WordLevel offsets only.

Failure modes to guard against:

- Added-token strip handling consumes the right whitespace span on a simple
  WordLevel path but fails once ByteLevel inserts prefix-space markers.
- `lstrip=true` emits `<mask>` without the consumed left space, reports offset
  `(8, 14)` instead of `(7, 14)`, or sends the consumed space through the BPE
  model as a separate token.
- `rstrip=true` emits `<mask>` without the consumed right space, drops the
  standalone `Ġ` before `<mask>`, or fails the `add_prefix_space=true` case
  where the following emoji token still carries a prefix-space marker.
- Byte-level BPE splits Unicode bytes but records offsets as byte fragments
  instead of mapping every byte-level piece back to the original Unicode scalar
  span, especially the `i⭢j` `(1, 4)` checks and the overlapping emoji offsets
  in `lstrip_tokens`.
- Offset trimming removes punctuation spans, shifts the first token because of
  synthetic prefix space, or trims only ASCII whitespace in the normalized view
  rather than reporting byte offsets into the original input.
- Tests pass by loading `ByteLevel` JSON tags without executing the production
  ByteLevel pre-tokenizer, BPE model, and post-processor together.

Verification commands:

- Upstream reference:
  `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && cargo test --test added_tokens`
- Current C++ added-token-adjacent smoke:
  `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
  `cmake --build projects/tokenizers.cpp/build`
  `ctest --test-dir projects/tokenizers.cpp/build -R tokenizers_cpp_added_tokens_test --output-on-failure`
- Full C++ parity suite:
  `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`

## QA Audit: Byte-Level Pair And Pre-Tokenized Offsets

Source audited:

- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_double_sequence`
- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_pre_tokenized_sequence`
- Shared upstream ByteLevel/GPT-2 fixture helper in
  `third_party/tokenizers/tokenizers/tests/common/mod.rs`
- Current C++ API, implementation, and parity tests in
  `include/tokenizers_cpp/tokenizer.hpp`, `src/tokenizer.cpp`, and
  `tests/parity`

Boundary spec: `byte-level-pair-pretokenized-offsets.md`.

Upstream fixture requirements:

- Both tests use `get_byte_level(add_prefix_space=true, trim_offsets=...)`,
  i.e. GPT-2 BPE from `data/gpt2-vocab.json` and `data/gpt2-merges.txt`, a
  `ByteLevel` pre-tokenizer with prefix spaces enabled, a default `ByteLevel`
  decoder, and a `ByteLevel` post-processor.
- `byte_level_double_sequence` encodes a pair with `add_special_tokens=false`.
  Offsets are relative to each original sequence, not concatenated global text.
- `byte_level_pre_tokenized_sequence` encodes a pre-tokenized word slice with
  `add_special_tokens=false`. Offsets are relative to each input word.

Exact acceptance from the upstream reference:

- `offsets.rs::byte_level_double_sequence`, untrimmed:
  - Inputs: `input_a = "My name is Anthony"`,
    `input_b = "What is my name?"`, `trim_offsets=false`.
  - Ids:
    `[2011,1438,318,9953,1867,318,616,1438,30]`.
  - Tokens:
    `["ĠMy","Ġname","Ġis","ĠAnthony","ĠWhat","Ġis","Ġmy","Ġname","?"]`.
  - Offsets:
    `[(0,2),(2,7),(7,10),(10,18),(0,4),(4,7),(7,10),(10,15),(15,16)]`.
  - Word ids:
    `[Some(0),Some(1),Some(2),Some(3),Some(0),Some(1),Some(2),Some(3),Some(4)]`.
  - Type ids:
    `[0,0,0,0,1,1,1,1,1]`.
  - Special-token mask:
    `[0,0,0,0,0,0,0,0,0]`.
  - Attention mask:
    `[1,1,1,1,1,1,1,1,1]`.
- `offsets.rs::byte_level_double_sequence`, trimmed:
  - Same inputs with `trim_offsets=true`.
  - Ids, tokens, word ids, type ids, special-token mask, and attention mask are
    the same as the untrimmed case.
  - Offsets:
    `[(0,2),(3,7),(8,10),(11,18),(0,4),(5,7),(8,10),(11,15),(15,16)]`.
- `offsets.rs::byte_level_pre_tokenized_sequence`:
  - Input words: `["My","name","is","Anthonino"]`,
    `trim_offsets=false`.
  - Ids:
    `[2011,1438,318,8451,261,2879]`.
  - Tokens:
    `["ĠMy","Ġname","Ġis","ĠAnth","on","ino"]`.
  - Offsets:
    `[(0,2),(0,4),(0,2),(0,4),(4,6),(6,9)]`.
  - Word ids:
    `[Some(0),Some(1),Some(2),Some(3),Some(3),Some(3)]`.
  - Type ids:
    `[0,0,0,0,0,0]`.
  - Special-token mask:
    `[0,0,0,0,0,0]`.
  - Attention mask:
    `[1,1,1,1,1,1]`.

Current C++ coverage comparison:

- `tests/parity/byte_level_bpe_test.cpp` covers the current single-sequence
  ByteLevel/BPE slice for `offsets.rs::{byte_level_basic,byte_level_unicode}`
  and `added_tokens.rs::{lstrip_tokens,rstrip_tokens}`. It also exercises the
  pair and pre-tokenized paths from this audit.
- `Tokenizer::encode_pair(text_a, text_b, ...)` exists and currently merges two
  string encodings, preserves second-sequence offsets relative to `text_b`,
  appends second-sequence word ids, and rewrites second-sequence type ids to
  `1`.
- `include/tokenizers_cpp/tokenizer.hpp` declares
  `encode(const std::vector<std::string>&, ...)` for pre-tokenized words, and
  `src/tokenizer.cpp` defines it through the production ByteLevel/BPE path.
- Existing single-sequence helper assertions check all-zero type ids,
  special-token masks, attention masks, and word ids for covered ByteLevel
  cases, but those checks should not be treated as pair or pre-tokenized
  coverage.
- Pair and pre-tokenized C++ coverage now pins exact tokens and exact id arrays
  in addition to offsets, word ids, type ids, special-token masks, and
  attention masks.

Failure modes to guard against:

- Pair offsets for sequence B are shifted by the byte length of sequence A
  instead of remaining relative to `input_b`.
- Pair word ids continue from sequence A instead of resetting to `Some(0)` for
  the first word of sequence B.
- Pair type ids remain all `0`, or masks are rebuilt with the wrong length after
  merging the two encodings.
- Trimmed pair processing incorrectly trims the synthetic prefix marker from
  the first token of sequence B; upstream keeps `ĠWhat` at offset `(0,4)`.
- Pre-tokenized offsets are reported against a joined string instead of each
  input word, causing `"name"` to appear as `(3,7)` or similar instead of
  `(0,4)`.
- BPE subpieces from one pre-tokenized word receive fresh word ids; upstream
  keeps `ĠAnth`, `on`, and `ino` on `Some(3)`.
- Tests accidentally cover the pre-tokenized overload through a shortcut path
  that bypasses production ByteLevel/BPE tokenization.

Verification commands:

- Upstream exact checks:
  `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_double_sequence -- --exact`
  `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets byte_level_pre_tokenized_sequence -- --exact`
- Upstream field probe used for the vectors above:
  temporary Rust binary under `projects/tokenizers.cpp/build/offset-probe`
  depending on the local upstream crate and local GPT-2 test data.
- Current C++ byte-level coverage smoke:
  `ctest --test-dir projects/tokenizers.cpp/build -R tokenizers_cpp_byte_level_bpe_test --output-on-failure`

## QA Audit: Split On Added Tokens Bert

Source audited:

- `third_party/tokenizers/tokenizers/tests/offsets.rs::split_on_added_tokens_bert`
- Shared upstream `get_bert()` fixture helper in
  `third_party/tokenizers/tokenizers/tests/common/mod.rs`
- Current C++ parity tests and docs in `tests/parity` and `docs/specs`

Boundary spec: `split-on-added-tokens-bert.md`.

Upstream fixture requirements:

- The tokenizer uses `get_bert()`: WordPiece from
  `data/bert-base-uncased-vocab.txt`, default `BertNormalizer`,
  `BertPreTokenizer`, default WordPiece decoder, and `BertProcessing`.
- The test then registers `[MASK]` as a special added token and calls
  `encode(input, false)`, so the encoded output must not include inserted
  `[CLS]` or `[SEP]` tokens.

Exact acceptance from the upstream reference:

- Input: `Yesterday I saw a [MASK] far away`.
- Ids:
  `[7483,1045,2387,1037,103,2521,2185]`.
- Tokens:
  `["yesterday","i","saw","a","[MASK]","far","away"]`.
- Offsets:
  `[(0,9),(10,11),(12,15),(16,17),(18,24),(25,28),(29,33)]`.
- Word ids:
  `[Some(0),Some(1),Some(2),Some(3),Some(4),Some(5),Some(6)]`.
- Type ids:
  `[0,0,0,0,0,0,0]`.
- Special-token mask:
  `[0,0,0,0,0,0,0]`.
- Attention mask:
  `[1,1,1,1,1,1,1]`.

Current C++ coverage comparison:

- `tests/parity/added_tokens_test.cpp` covers added-token id assignment,
  upstream all-zero encode special masks, decode skipping, single-word
  matching, overlap precedence, and lstrip/rstrip span capture on
  WordLevel-oriented paths.
- `tests/parity/json_wrapper_dispatch_test.cpp` accepts `BertNormalizer`,
  `BertPreTokenizer`, `BertProcessing`, and `WordPiece` JSON tags as dispatch
  coverage, but does not prove their runtime behavior.
- `tests/parity/serialization_model_load_test.cpp` loads a realistic WordPiece
  vocab and checks representative id lookups.
- `tests/parity/bert_wordpiece_added_tokens_test.cpp` constructs the
  upstream-equivalent BERT tokenizer JSON from local test data and asserts the
  exact `[MASK]` split ids, tokens, offsets, word ids, type ids,
  special-token mask, and attention mask.

Failure modes to guard against:

- `[MASK]` is lowercased or normalized before added-token matching and no longer
  matches the special token.
- `encode(input, false)` inserts `[CLS]` or `[SEP]` despite upstream disabling
  added post-processor tokens for this call.
- Offsets are reported against the normalized/lowercased string instead of the
  original input.
- The added special token receives `word_id = null`, wrong type id, or
  `special_tokens_mask = 1`; upstream keeps the mask all zero because no
  post-processor special tokens are inserted.
- Dispatch-only support for `BertPreTokenizer` or `WordPiece` is mistaken for
  runtime parity.

Verification commands:

- Upstream exact check:
  `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets split_on_added_tokens_bert -- --exact`
- Current C++ parity suite after implementation:
  `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build -DTOKENIZERS_CPP_BUILD_TESTS=ON -DTOKENIZERS_CPP_FETCH_DEPS=OFF`
  `cmake --build projects/tokenizers.cpp/build`
  `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`

## R1 Model JSON Load Acceptance

See `model-json-load.md` for the dependency boundary, load acceptance rules,
assertions to add now, and behavior blockers. This slice is native C++ JSON
deserialization coverage only: no Rust FFI, runtime shell-out, training,
HTTP/from-pretrained, benches, wrappers, tokenization behavior claims, or edits
under `third_party/tokenizers`.

## R1 JSON Wrapper Dispatch Acceptance

The next target ports the dispatch-sensitive parts of
`serialization.rs::{normalizers,processors,pretoks,decoders,models,tokenizer}`. Because
ADR-0002 does not require C++ JSON serialization parity yet, C++ tests should
use the upstream serialized JSON strings as load inputs and structured
assertions, not as a requirement to re-emit byte-identical JSON.

See `serialization-wrapper-dispatch.md` for the dispatch-only versus
behavior-required split for this slice.

Required positive dispatch coverage:

- `normalizers`: load `{"type":"NFC"}` as `NFC`, and load the default
  `BertNormalizer` JSON with `clean_text`, `handle_chinese_chars`,
  `strip_accents`, and `lowercase` preserved.
- `pretoks`: load `{"type":"BertPreTokenizer"}`,
  `{"type":"CharDelimiterSplit","delimiter":" "}`,
  `{"type":"Whitespace"}`, and both `Split` pattern forms:
  `{"String":"[SEP]"}` and `{"Regex":"[SEP]"}` with behavior `Isolated` and
  `invert:false`.
- `processors`: load exact upstream `BertProcessing` serde shape
  `{"type":"BertProcessing","sep":["SEP",0],"cls":["CLS",0]}`.
- `decoders`: load default `ByteLevel` decoder JSON with
  `add_prefix_space:true`, `trim_offsets:true`, and `use_regex:true`.
- `models`: load default `BPE` model wrapper JSON through the model slot,
  including required default fields that upstream emits.
- `tokenizer`: load a full tokenizer JSON containing a default `WordPiece`
  model and `NFC` normalizer, proving the top-level loader accepts those exact
  tags through the normal tokenizer load path.

Status: implemented in `tests/parity/json_wrapper_dispatch_test.cpp` as a
dispatch-only C++ test. It validates load success/failure through
`Tokenizer::from_file`; it does not prove component behavior.

Required negative dispatch coverage:

- Because C++ has no public typed generic tokenizer API, upstream wrong
  concrete-type assertions must be represented as wrong-slot or internal parser
  validation. Do not add public introspection only for these tests.
- A wrapper slot must reject unsupported `type` strings with an error that
  names the component slot and offending type.
- `Tokenizer::from_file` must reject malformed known component fields instead
  of silently dropping the component or replacing it with a no-op.

Known gaps that should remain visible during review:

- Model behavior parity for BPE and Unigram is not proven by these dispatch
  tests. The model JSON load slice covers load integrity only; later
  encode/decode tests must cover remaining runtime behavior.
- C++ serialization output parity is intentionally out of scope for this target.
