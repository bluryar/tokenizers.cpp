# Serialization Wrapper Dispatch Slice

## Source

Upstream target:
`third_party/tokenizers/tokenizers/tests/serialization.rs::{normalizers,processors,pretoks,decoders,models,tokenizer}`
at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`.

This slice translates upstream serde wrapper dispatch into C++ JSON load tests.
It does not require byte-for-byte serialization because `tokenizers.cpp` has no
serialization API and ADR-0002 makes load and inference parity the product goal.

## C++ Tests To Add Now

C++ parity tests should use local tokenizer JSON fixtures and call
`Tokenizer::from_file`. They should assert structured load success or failure,
not pretty-printed JSON equality and not Rust generic type behavior.

- `serialization.rs::normalizers`: accept `normalizer` wrappers with exact
  tags `NFC` and `BertNormalizer` using upstream field names. Do not require
  NFC or Bert normalization output yet.
- `serialization.rs::pretoks`: accept `pre_tokenizer` wrappers with exact tags
  `BertPreTokenizer`, `CharDelimiterSplit`, `Whitespace`, and `Split`, including
  `Split` patterns `{"String":"[SEP]"}` and `{"Regex":"[SEP]"}`. Do not require
  pre-tokenization output yet.
- `serialization.rs::processors`: accept the `post_processor` wrapper with
  exact tag `BertProcessing` and the upstream serde shape
  `{"type":"BertProcessing","sep":["SEP",0],"cls":["CLS",0]}`. Do not require
  post-processing output yet.
- `serialization.rs::decoders`: accept a `decoder` wrapper with exact tag
  `ByteLevel` and upstream default fields. Do not require ByteLevel decode
  behavior yet.
- `serialization.rs::models`: accept a `model` wrapper with exact tag `BPE` and
  the upstream default JSON shape. Do not require BPE tokenization yet.
- `serialization.rs::tokenizer`: accept a full tokenizer JSON object containing
  `version`, null optional slots, a `WordPiece` model wrapper, and an `NFC`
  normalizer wrapper. This test proves top-level graph loading, not WordPiece
  tokenization.

Negative tests should cover an unknown `type` in each implemented slot and a
known type in the wrong slot. The assertion is that `Tokenizer::from_file`
throws and the diagnostic names the slot and offending type.

## Dispatch-Only Versus Behavior-Required

Dispatch-only for this slice means the loader recognizes the component tag,
validates required fields and enum values, stores the component config in the
runtime graph, and rejects malformed values. Tests must not encode or decode
through these components as proof of behavior yet.

Dispatch-only component tags for this slice:

- Normalizers: `NFC`, `BertNormalizer`.
- Pre-tokenizers: `BertPreTokenizer`, `CharDelimiterSplit`, `Split`.
- Post-processors: `BertProcessing`.
- Decoders: `ByteLevel`.
- Models: `BPE`, `WordPiece`.

`Whitespace` started as dispatch-only for this slice, but it now has native
runtime coverage in the real `documentation.rs::pipeline_bert` parity fixture.

Behavior-required now is limited to the behavior already covered by existing C++
tests: top-level tokenizer JSON parsing, `WordLevel` vocab/id lookup, added
token `content`/`id`/`special` loading, simple whitespace token splitting for
the existing smoke fixture, and simple decode of known ids. New wrapper dispatch
tests should not broaden model, Unicode, regex, byte-level, or decoder behavior.

## Unsupported Or Incomplete Types

For non-null component objects, missing `type`, unsupported `type`, malformed
required fields, invalid enum values, and a valid tag placed in the wrong
component slot are load errors. Optional slots may be absent or `null`; those are
the only no-op forms.

An accepted dispatch-only component that reaches encode/decode before its
behavior is implemented may fail with a not-implemented diagnostic that names
the component. Later behavior parity tests must replace that with upstream
semantics; dispatch tests must not weaken expected encoding fields.

## Remaining Blockers

- No public C++ serialization or typed generic tokenizer API exists, so upstream
  serde roundtrip and wrong-generic deserialization assertions are represented
  only by load acceptance/failure.
- Unicode normalization, regex split behavior, byte-level behavior,
  WordPiece/BPE/Unigram inference, post-processor behavior, truncation,
  padding, and decoder semantics are covered by later runtime specs and the
  core inference completion statement.
- Component introspection should not be added to the public API only for these
  tests; use load success/failure and later behavior fixtures instead.
