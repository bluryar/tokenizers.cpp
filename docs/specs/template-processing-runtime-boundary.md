# TemplateProcessing Runtime Boundary

## Source

Semantic references:

- `third_party/tokenizers/tokenizers/src/processors/template.rs`
- `third_party/tokenizers/tokenizers/src/tokenizer/encoding.rs`

This slice remains native C++ inference only. It must not add Rust FFI, runtime
shell-out, trainers, HTTP/from-pretrained loading, or edits under
`third_party/tokenizers`.

## Covered Behavior

The native tokenizer now executes serialized `TemplateProcessing` piece arrays
through the shared post-processing path for both direct post-processors and
post-processor `Sequence` children.

Covered runtime behavior:

- `Sequence` pieces for `A` and `B`, including non-default type ids and pair
  templates that place `B` before `A`.
- `SpecialToken` pieces with one or more ids/tokens.
- Special-token type ids, offsets `(0, 0)`, null word ids, special-token masks,
  and attention masks.
- `add_special_tokens=false`: special pieces are skipped, while sequence order
  and sequence type-id overrides from the template are still applied.
- Truncation accounting for multi-id special-token pieces before template
  insertion.
- Pair overflowing cross-products: first-side overflow, both-side overflow, and
  second-side overflow are exposed at the top level, and the both-side overflow
  is also nested under the corresponding one-sided overflowing encodings.

## Current C++ Coverage

`tests/parity/template_processing_test.cpp` covers:

- A broad direct `TemplateProcessing` fixture with a two-token special bundle,
  pair templates ordered as `special + B + special + A + special`, and type ids
  `9/5/6/2/8`.
- `add_special_tokens=false` for single and pair encodes, verifying that
  sequence order and type ids still come from the template.
- Single-sequence truncation with a multi-id special bundle.
- Pair truncation/overflowing with TemplateProcessing, including nested
  both-overflow encodings.

Existing coverage also includes:

- `tests/parity/sentencepiece_albert_test.cpp` for ALBERT/BERT-shaped
  TemplateProcessing.
- `tests/parity/roberta_processing_test.cpp` for RoBERTa-style special-token
  insertion.
- `tests/parity/post_processor_sequence_test.cpp` for
  `Sequence(ByteLevel, TemplateProcessing)` composition.
- `tests/parity/llama_encode_test.cpp` for real local Llama template insertion.

## Known Gaps

- Builder/string template forms are a Rust construction API surface and are not
  accepted as tokenizer JSON. The accepted C++ surface is serialized piece-array
  JSON.
- Broader real-tokenizer smoke should continue to exercise uncommon template
  shapes as they appear in local fixtures.
- Sequence range introspection is not exposed in the public C++ `Encoding` API;
  parity is asserted through ids, tokens, offsets, word ids, type ids, masks,
  attention masks, and overflowing encodings.
