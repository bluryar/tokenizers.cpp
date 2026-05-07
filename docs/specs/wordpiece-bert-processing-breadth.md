# WordPiece And Bert Processing Breadth

## Source

Semantic references:

- `third_party/tokenizers/tokenizers/src/models/wordpiece/mod.rs`
- `third_party/tokenizers/tokenizers/src/decoders/wordpiece.rs`
- `third_party/tokenizers/tokenizers/src/normalizers/bert.rs`
- `third_party/tokenizers/tokenizers/src/normalizers/utils.rs`
- `third_party/tokenizers/tokenizers/src/pre_tokenizers/bert.rs`
- `third_party/tokenizers/tokenizers/src/processors/bert.rs`

This slice extends the native C++ BERT path beyond
`offsets.rs::split_on_added_tokens_bert` without widening project scope: no
Rust FFI, no runtime shell-out, no training, no HTTP/from-pretrained loading,
and no edits under `third_party/tokenizers`.

## Covered Behavior

WordPiece model inference now has C++ parity coverage for:

- Greedy longest-match subword splitting on UTF-8 character boundaries.
- `continuing_subword_prefix` application for non-initial subwords.
- `[UNK]` fallback when no full segmentation exists.
- `[UNK]` fallback when `max_input_chars_per_word` is exceeded.
- Original byte offsets and shared `word_id` across subwords.
- Serialized simple normalizer sequences before WordPiece when the tokenizer
  JSON uses `Sequence(NFD, Lowercase, StripAccents)` instead of a direct
  `BertNormalizer`, as in the local `bert-wiki.json` fixture.
- Direct serialized `Whitespace` pre-tokenizer runtime via the shared ICU regex
  backend, matching upstream's `\w+|[^\w\s]+` grouping for the accepted BERT
  documentation pipeline fixture.

`BertProcessing` now has C++ parity coverage for:

- Single sequence `add_special_tokens=true`: `[CLS] $A [SEP]`.
- Pair sequence `add_special_tokens=true`: `[CLS] $A [SEP] $B:1 [SEP]:1`.
- Pair sequence `add_special_tokens=false`: raw pair concatenation with second
  sequence `type_id=1`.
- Inserted special-token offsets `(0, 0)`, `word_id=null`,
  `special_tokens_mask=1`, and attention mask entries.
- Real local `bert-wiki.json` with injected truncation and fixed padding over
  raw and pre-tokenized single/pair encodings plus ordered batch/pair-batch
  APIs, including overflowing subword encodings.
- Explicit char-offset overloads over the same real BERT matrix, including
  recursive conversion of overflowing subword encodings and word-relative
  pre-tokenized char offsets.

`BertNormalizer` now has C++ parity coverage for the current inference subset:

- `clean_text` removal of common control/format characters, including
  zero-width space, while preserving original byte-span projection.
- Unicode whitespace and control cleanup routed through the shared Unicode
  backend. The default vendored ICU build uses ICU categories.
- Chinese-character spacing before `BertPreTokenizer`.
- Default `strip_accents = lowercase` behavior using backend NFD normalization
  and nonspacing-mark classification.
- Lowercase after accent stripping through the shared Unicode backend.
- Default-ICU Rust-derived fixture coverage for `Ἀ he\u{E000}llo` normalizing
  and tokenizing as `α hello`, including original byte offsets.

`BertPreTokenizer` now has C++ parity coverage for the current punctuation
subset:

- Whitespace splitting uses the same covered Unicode whitespace table as
  `BertNormalizer` cleanup, so `BertPreTokenizer` also behaves sensibly when
  used without a normalizer.
- Punctuation splitting routes through the shared Unicode backend. The default
  vendored ICU build uses Unicode punctuation categories.
- Common Unicode punctuation is isolated on UTF-8 codepoint boundaries,
  including CJK/fullwidth punctuation, general punctuation used by the local
  BERT fixture, and a default-ICU supplementary-plane punctuation fixture.

`WordPiece` decoder now has C++ parity coverage for the current inference
subset:

- Decoder JSON `prefix` and `cleanup` fields are parsed into runtime state.
- Runtime `with_wordpiece_decoder(prefix, cleanup)` replaces the active decoder
  chain for the accepted documentation-style BERT pipeline mutation.
- Continuation prefixes are removed only for non-initial tokens, matching
  upstream behavior where an initial `##token` remains prefixed.
- `cleanup=true` applies upstream's focused artifact cleanup rules for spaces
  before punctuation and common contractions.
- `cleanup=false` preserves raw decoder spacing and prefix behavior.
- With no serialized decoder, WordPiece `decode(ids, skip_special_tokens=true)`
  joins the remaining tokens with spaces, preserving continuation prefixes as
  upstream does before an explicit WordPiece decoder is installed.

The C++ coverage lives in
`tests/parity/bert_wordpiece_added_tokens_test.cpp` and uses the local
`bert-base-uncased-vocab.txt` fixture, plus a small custom WordPiece vocab for
Greek/Cyrillic lowercase and decoder cleanup behavior.
It also loads the real local `bert-wiki.json` fixture and injects temporary
truncation/padding settings for Rust-derived API-matrix smoke vectors.
The same real fixture also covers `documentation.rs::pipeline_bert`, including
`[CLS] welcome to the [UNK] tok ##eni ##zer ##s library . [SEP]`, byte offsets,
word ids, no-decoder token joining, and runtime WordPiece decoder cleanup.

## Remaining Gaps

- Full Unicode category/database parity for `BertNormalizer` depends on the
  default vendored ICU backend.
- Full Unicode punctuation category parity for `BertPreTokenizer` depends on
  the default vendored ICU backend.
- Sequence ranges remain a follow-up. Direct serialized TemplateProcessing and
  tokenizer-level truncation, padding, and overflowing encodings are covered by
  later slices; this BERT slice only claims the real `bert-wiki.json` API
  matrix described above.
