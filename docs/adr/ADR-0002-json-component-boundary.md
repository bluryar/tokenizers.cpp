# ADR-0002: Tokenizer JSON And Component Boundary

## Status

Accepted

## Context

`tokenizers.cpp` must load local Hugging Face tokenizer JSON files and execute
the inference pipeline in native C++. The Rust crate remains the semantic
reference, but runtime code must not call Rust, Python, HTTP APIs, or training
surfaces.

## Decision

### Tokenizer JSON

The C++ loader must treat the tokenizer JSON object as the runtime graph. The
supported top-level fields are:

- `version`: when present, must be `"1.0"`; other versions are load errors.
- `truncation`: absent or `null` means disabled; otherwise truncation params.
- `padding`: absent or `null` means disabled; otherwise padding params.
- `added_tokens`: absent means empty; otherwise added-token records with stable
  ids and flags.
- `normalizer`: absent or `null` means none; otherwise a normalizer component
  wrapper.
- `pre_tokenizer`: absent or `null` means none; otherwise a pre-tokenizer
  component wrapper.
- `post_processor`: absent or `null` means none; otherwise a post-processor
  component wrapper.
- `decoder`: absent or `null` means none; otherwise a decoder component wrapper.
- `model`: required inference model wrapper.

Unknown top-level fields are ignored for compatibility, matching upstream
runtime deserialization. Known fields with malformed values are load errors.
Serialization parity is not a product goal yet; load and inference parity are.

### Wrapper Dispatch

Component wrappers dispatch from the exact Hugging Face serialized `type`
string and preserve upstream field names. The first supported dispatch set is:

- Models: `BPE`, `WordPiece`, `WordLevel`, `Unigram`.
- Normalizers: `BertNormalizer`, `Strip`, `StripAccents`, `NFC`, `NFD`,
  `NFKC`, `NFKD`, `Sequence`, `Lowercase`, `Nmt`, `Precompiled`, `Replace`,
  `Prepend`, `ByteLevel`.
- Pre-tokenizers: `BertPreTokenizer`, `ByteLevel`, `CharDelimiterSplit`,
  `Metaspace`, `Whitespace`, `Sequence`, `Split`, `Punctuation`,
  `WhitespaceSplit`, `Digits`, `UnicodeScripts`, `FixedLength`.
- Post-processors: `RobertaProcessing`, `BertProcessing`, `ByteLevel`,
  `TemplateProcessing`, `Sequence`.
- Decoders: `BPEDecoder`, `ByteLevel`, `WordPiece`, `Metaspace`, `CTC`,
  `Sequence`, `Replace`, `Fuse`, `Strip`, `ByteFallback`.

Legacy untagged model JSON may be accepted only where upstream Rust accepts it.
For ambiguous legacy forms, preserve upstream probe order: `WordPiece` before
`WordLevel`, and `RobertaProcessing` before `BertProcessing`. New C++ fixtures
should prefer explicit `type` fields.

Unsupported component `type` values fail during `Tokenizer::from_file` with a
diagnostic that names the component slot and type. A component may not silently
degrade to a no-op unless upstream treats the exact JSON value as `null`.

### Ownership And Runtime Order

`Tokenizer` owns the model, optional components, added vocabulary, truncation,
and padding after JSON load. The JSON document and parsed `nlohmann::json`
values must not be retained as the primary runtime representation.

The inference order is:

1. Load `model`, optional components, truncation, padding, then add
   `added_tokens` using the loaded normalizer state.
2. Encode input text by running added-token extraction and normalization,
   pre-tokenization, and model tokenization.
3. For single and pair encodings, run truncation before post-processing so the
   truncation length accounts for special tokens, then run post-processing and
   padding in upstream runtime order.
4. Preserve encoding fields exactly: `ids`, `type_ids`, `tokens`, `offsets`,
   `word_ids`, `special_tokens_mask`, `attention_mask`, and `overflowing`.
5. Decode ids by resolving added tokens plus model vocabulary, honoring
   `skip_special_tokens`, then running the decoder component when present.

Sequence components own ordered child component lists and execute children in
JSON order. Offset ranges are byte offsets into the original input unless an
upstream test explicitly proves a different referential for that field.

### Runtime Exclusions

The native runtime excludes trainers, training JSON, sampling-only behavior,
HTTP or `from_pretrained` loading, network access, Python/Node/WASM wrappers,
benchmarks, Rust FFI, and runtime shell-out. Excluded upstream tests must remain
classified in docs as `skip-training`, `skip-non-core`, or `reference-only`.

### Dependency Use

Use the pinned dependencies from ADR-0001 only when a component needs them:

- Local `include/nlohmann/json.hpp`: JSON parsing and validation. This must be a
  copied single-header dependency in `tokenizers.cpp`, not `FetchContent` or
  network access during CMake configure.
- Unicode normalization and category behavior must not rely on system dynamic
  libraries. Current covered slices use native C++ code; ICU4C adoption must
  follow `ADR-0003`: pinned, project-owned, and capable of a production static
  build with explicit ICU data packaging.
- Regex-compatible split and replace behavior must not rely on system dynamic
  libraries. Any future RE2 adoption must be vendored/static or explicitly
  optional.

Do not add a dependency for trainer-only features. Do not vendor or patch
`third_party/tokenizers`.

### Fixtures

C++ parity fixtures must be local files, deterministic, and small enough to
review. Fixture metadata must record:

- upstream git ref,
- upstream Rust test or generator command,
- tokenizer JSON path or inline JSON source,
- input text or pair text,
- encode/decode options,
- expected output JSON.

Expected outputs should be generated from upstream Rust tests or a local dev
generator that runs against `third_party/tokenizers` outside the C++ runtime.
Committed fixtures may include GPT/BERT/Roberta/SentencePiece-style tokenizer
JSON excerpts, but not downloaded-at-test-time artifacts.

### Error Behavior

`Tokenizer::from_file` fails on unreadable files, invalid JSON, unsupported
`version`, missing `model`, unsupported component `type`, malformed required
component fields, invalid ids, invalid merges, and truncation/padding parameter
errors. Runtime encode/decode fails on invalid inputs that upstream reports as
errors, such as unresolved ids when a test covers that behavior.

Error messages do not need byte-for-byte Rust parity. They must identify the
operation and offending field or component well enough for a failing fixture to
be actionable.

### C++ Parity Tests

C++ parity tests compare structured results, not pretty-printed JSON. For every
ported upstream runtime test, compare all fields exposed by the upstream test or
fixture:

- `ids`
- `type_ids`
- `tokens`
- `offsets`
- `word_ids`
- `special_tokens_mask`
- `attention_mask`
- `overflowing`
- decoded text

Offsets must be compared as exact integer pairs. Masks must be compared as exact
integer arrays. Decode tests must cover both `skip_special_tokens=true` and
`false` when the upstream test distinguishes them. Pair encoding tests must
check type ids, special-token masks, and overflow ordering.

When upstream Rust behavior is intentionally excluded, the C++ test should assert
the documented exclusion or omit the runtime case with a link to the inventory.
When upstream Rust behavior is accepted but not implemented yet, the C++ test
should be marked as a known blocker in docs rather than weakened.

## Consequences

Implementation can proceed component by component without revisiting the JSON
graph boundary. Parity failures should now classify as one of: loader dispatch,
component semantics, offset tracking, truncation/padding, decode behavior, or an
explicit documented exclusion.
