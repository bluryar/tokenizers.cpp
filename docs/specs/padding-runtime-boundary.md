# Padding Runtime Boundary

## Scope

This slice ports the native C++ inference path for tokenizer padding after
post-processing.

Covered behavior:

- tokenizer JSON `padding` load with:
  - `strategy`
  - `direction`
  - `pad_to_multiple_of`
  - `pad_id`
  - `pad_type_id`
  - `pad_token`
- `PaddingStrategy::Fixed(size)`
- `PaddingStrategy::BatchLongest` for public single-sequence and pair batch
  APIs
- `PaddingDirection::{Right,Left}`
- right and left padding over all `Encoding` fields
- `pad_to_multiple_of`, including fixed-size rounding
- recursive padding of `Encoding::overflowing`
- padding after single/pair post-processing
- padding after single/pair truncation and overflow merge
- fixed padding over real tokenizer JSON smoke paths after ByteLevel/BPE and
  post-processor execution
- fixed padding over real `bert-wiki.json` after serialized simple
  normalization, Whitespace pre-tokenization, WordPiece, and TemplateProcessing
- fixed padding over real Llama JSON smoke paths after
  `Sequence(Split, ByteLevel(use_regex=false))`, BPE, and template insertion,
  including public batch and pair-batch output
- public direct pre-tokenized pair overload:
  `encode_pair(vector<string>, vector<string>)`
- public `Tokenizer::encode_batch(...)` for single sequences
- public `Tokenizer::encode_batch_pairs(...)` for pair sequences
- public pre-tokenized batch overloads:
  `encode_batch(vector<vector<string>>)` and
  `encode_batch_pairs(vector<pair<vector<string>, vector<string>>>)`

Excluded behavior:

- padding sequence range adjustments, because C++ `Encoding` does not expose
  sequence ranges yet
- dynamic padding policy interactions outside the current public encode APIs

## Accepted Fixtures

`tests/parity/padding_test.cpp` is the native acceptance surface for this
slice.

It verifies:

- fixed right padding for a single sequence:
  - ids/tokens append `[PAD]`
  - offsets become `(0, 0)`
  - word ids become `null`
  - special-token mask is `1`
  - attention mask is `0`
- fixed left padding preserves original token order after pad tokens and uses
  configured `pad_type_id`
- fixed padding rounded by `pad_to_multiple_of`
- padding applies recursively to single-sequence overflowing encodings
- `BatchLongest` targets the main encoding length for a single public encode and
  still pads shorter overflowing encodings
- `BatchLongest` targets the maximum top-level encoding length across a batch of
  single sequences
- `BatchLongest` plus `pad_to_multiple_of` rounds the shared batch target
- `BatchLongest` targets the maximum top-level encoding length across a batch of
  pair sequences
- overflowing encodings inside a batch are padded to the same shared batch
  target
- pre-tokenized single-sequence batch encoding with truncation, overflowing,
  per-word offsets, word ids, and shared `BatchLongest` padding
- direct pre-tokenized pair encoding with pair truncation, overflow merge
  order, fixed padding, type ids, and word ids
- pre-tokenized pair batch encoding with pair truncation, overflow merge order,
  type ids, word ids, and shared `BatchLongest` padding
- fixed pair padding after Bert/ALBERT-shaped special-token insertion
- fixed pair padding after pair truncation and overflow merge
- invalid padding direction is rejected during JSON load
- real local `roberta.json` with injected truncation and fixed padding, where
  main and overflowing encodings are padded after RoBERTa special-token
  insertion, including pre-tokenized single/pair batch variants
- real local `bert-wiki.json` with injected truncation and fixed padding, where
  main and overflowing encodings are padded after TemplateProcessing
  `[CLS]/[SEP]` insertion, including raw/pre-tokenized single, pair, batch, and
  pair-batch variants
- real local GPT-style `tokenizer.json` with injected truncation and fixed
  padding over `Sequence(ByteLevel, TemplateProcessing)`
- real local `llama-3-tokenizer.json` with injected truncation and fixed
  padding, using `<|finetune_right_pad_id|>` over single, pair, batch, and
  pair-batch encodings, including pre-tokenized batch variants with
  word-relative offsets

## Upstream References

Reference behavior comes from:

- `tokenizer::encoding::tests::padding`
- `utils::padding::tests::pad_to_multiple`
- `documentation.rs::quicktour` padding example

## Implementation Notes

Padding runs after truncation and post-processing, matching upstream
`TokenizerImpl::post_process`.

All vectors are padded together:

- `ids`
- `type_ids`
- `tokens`
- `offsets`
- `word_ids`
- `special_tokens_mask`
- `attention_mask`

`Encoding::overflowing` is padded before the parent encoding, mirroring upstream
`Encoding::pad`.

For public batch APIs, `BatchLongest` computes the target from top-level
encodings in the returned batch, then pads each top-level encoding and each of
its overflowing encodings to that same target. Overflowing lengths do not raise
the batch target, matching upstream `pad_encodings` behavior.

Pre-tokenized direct and batch offsets follow the existing C++ pre-tokenized
single-input contract: offsets are relative to each supplied word, and pair
word ids remain relative to their own sequence.
