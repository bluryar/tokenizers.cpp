# Truncation Runtime Boundary

## Scope

This slice ports the native C++ inference path for tokenizer truncation:
single and pair truncation before post-processing, including stride-based
overflowing encodings and upstream-style pair overflow merge order.

Covered behavior:

- tokenizer JSON `truncation` load with `max_length`, `strategy`, `stride`, and
  optional `direction`
- old JSON compatibility where missing `direction` defaults to `Right`
- `TruncationDirection::{Right,Left}`
- `TruncationStrategy::LongestFirst` for single and pair sequences
- `TruncationStrategy::{OnlyFirst,OnlySecond}` for pair sequences
- `OnlySecond` single-sequence truncation failure when truncation is required
- `Encoding::overflowing` slices for right/left truncation and stride overlap
- pair overflow merge ordering:
  - first overflow + second main
  - first overflow + second overflow
  - first main + second overflow
- effective max length reduction for single-sequence post-processors that add
  special tokens, based on the compiled post-processing template
- effective max length reduction for pair post-processors that add special
  tokens, including the BERT/ALBERT three-token shape and the RoBERTa four-token
  shape
- special-token insertion over overflowing single and pair encodings
- truncation over real tokenizer JSON smoke paths before ByteLevel/RoBERTa or
  TemplateProcessing special-token insertion

Excluded behavior:

- padding
- batch padding/truncation policy
- arbitrary multi-special post-processor composition
- overflow merge behavior beyond the current two-sequence public API

## Accepted Fixtures

`tests/parity/truncation_test.cpp` is the native acceptance surface for this
slice.

It verifies:

- Right truncation to 4 tokens with stride 2:
  - main ids: `[1, 2, 3, 4]`
  - overflowing ids: `[3, 4, 5]`
- Left truncation to 3 tokens:
  - main ids: `[3, 4, 5]`
  - overflowing ids: `[1, 2]`
- old truncation JSON with no `direction` field defaults to Right
- invalid `stride >= max_length` is rejected on load
- `OnlySecond` single-sequence truncation fails when a second sequence is
  required but absent
- Pair `LongestFirst` truncation preserves upstream length allocation and
  overflow merge order
- Pair `OnlyFirst` and `OnlySecond` truncate only the requested side, preserving
  the other sequence in main and overflowing outputs
- Pair target-too-short errors when the requested side cannot shrink enough
- single-sequence special-token accounting:
  - `max_length=5` with BertProcessing leaves 3 content tokens
  - both main and overflowing encodings receive `[CLS]`/`[SEP]`
- pair special-token accounting:
  - `max_length=7` with BertProcessing leaves 4 total content tokens
  - main and overflowing outputs receive `[CLS] A [SEP] B [SEP]`
- real local `roberta.json` with injected truncation, where the overflow slice
  receives RoBERTa `<s>` / `</s>` and subsequent fixed padding
- real local GPT-style `tokenizer.json` with injected truncation, where the
  overflow slice receives the serialized `TemplateProcessing` special token and
  subsequent fixed padding
- real local `llama-3-tokenizer.json` with injected truncation, where the
  overflow slice receives the serialized Llama `<|begin_of_text|>` template
  special token and subsequent fixed padding in single, pair, batch, and
  pair-batch outputs

Every assertion checks ids, tokens, offsets, word ids, type ids,
special-token masks, attention masks, and overflowing content where applicable.

## Upstream References

Reference behavior comes from:

- `tokenizer::encoding::tests::truncate_overflow_with_stride`
- `tokenizer::encoding::tests::truncate_left`
- `tokenizer::tests::{right_truncation_early_exit_matches_full_encode,left_truncation_keeps_tail_tokens}`
- `tokenizer::tests::{pair_right_truncation_longest_first,pair_only_second_does_not_truncate_first,pair_only_first_does_not_truncate_second}`
- `utils::truncation::tests::{truncate_encodings_longest_first,truncate_encodings_empty}`
- `utils::truncation::tests::test_deserialize_defaults`

## Implementation Notes

The C++ runtime slices all `Encoding` vectors together:

- `ids`
- `type_ids`
- `tokens`
- `offsets`
- `word_ids`
- `special_tokens_mask`
- `attention_mask`

Overflowing encodings are materialized as independent `Encoding` values with no
nested overflows. Public pair encoding calls the internal single-sequence
encoder with truncation disabled, then applies pair truncation once over the two
raw encodings. This mirrors upstream `truncate_encodings` and avoids silently
truncating each side independently.
