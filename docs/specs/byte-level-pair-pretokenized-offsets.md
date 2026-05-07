# Byte-Level Pair And Pre-Tokenized Offsets Boundary

## Source

Upstream targets at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_double_sequence`
- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_pre_tokenized_sequence`

This slice is native C++ inference runtime only. It must not add Rust FFI,
runtime shell-out, trainers, training fixtures, HTTP/from-pretrained loading,
wrapper APIs, or edits under `third_party/tokenizers`.

## Native C++ Boundary

Pair input should use the existing public boundary:

- `Tokenizer::encode_pair(text_a, text_b, add_special_tokens=true)`

Pre-tokenized single input should use one conservative overload:

- `Tokenizer::encode(const std::vector<std::string>& words, add_special_tokens=true)`

where `words` is a sequence of strings supplied by the caller. A later
tokenizer-level API slice also exposes direct and batch pre-tokenized pair
overloads through the same internal dispatcher. Do not add a public
`PreTokenizedString`, generic `InputSequence` variant, tokenizer mutation API,
or Rust-style builder surface for this slice.

Pre-tokenized semantics:

- Each supplied word is encoded as its own upstream subsequence.
- Added-token extraction, normalization, ByteLevel pre-tokenization, and BPE
  tokenization run independently per supplied word.
- `word_ids` are forced to the supplied word index for every model token
  produced from that word.
- Offsets are byte offsets into the supplied word, not into a concatenated
  string and not into a joined string with synthetic separators.
- With `ByteLevel(add_prefix_space=true)`, prefix-space behavior applies to
  each supplied word independently.

Pair semantics:

- Encode each raw sequence independently, then merge into one `Encoding`.
- Offsets for sequence B remain byte offsets into `text_b`; do not shift them
  by the length of `text_a`.
- `word_ids` restart from `0` for sequence B, matching upstream.
- `type_ids` are `0` for sequence A tokens and `1` for sequence B tokens.

The accepted tests call encode with `add_special_tokens=false`. Template
post-processors, special-token insertion for pairs, truncation, padding, and
overflowing encodings remain outside this slice.

## Fields To Compare

Every C++ parity fixture for these two upstream tests must compare exact:

- `ids`
- `tokens`
- `offsets`
- `word_ids`
- `type_ids`
- `special_tokens_mask`
- `attention_mask`

Here, "masks" means both `special_tokens_mask` and `attention_mask`. Compare
offsets as exact integer pairs and masks as exact integer arrays. Do not weaken
these tests to substring-only offset checks.

## Exact Acceptance

All ids below are GPT-2 BPE ids from `data/gpt2-vocab.json`.

### `byte_level_double_sequence`, `trim_offsets=false`

Fixture:

- tokenizer: `get_byte_level(add_prefix_space=true, trim_offsets=false)`
- input A: `"My name is Anthony"`
- input B: `"What is my name?"`
- encode option: `add_special_tokens=false`

Expected fields:

- `ids`: `[2011, 1438, 318, 9953, 1867, 318, 616, 1438, 30]`
- `tokens`:
  `["ĠMy", "Ġname", "Ġis", "ĠAnthony", "ĠWhat", "Ġis", "Ġmy", "Ġname", "?"]`
- `offsets`:
  `[(0, 2), (2, 7), (7, 10), (10, 18), (0, 4), (4, 7), (7, 10), (10, 15), (15, 16)]`
- `word_ids`:
  `[0, 1, 2, 3, 0, 1, 2, 3, 4]`
- `type_ids`: `[0, 0, 0, 0, 1, 1, 1, 1, 1]`
- `special_tokens_mask`: `[0, 0, 0, 0, 0, 0, 0, 0, 0]`
- `attention_mask`: `[1, 1, 1, 1, 1, 1, 1, 1, 1]`

### `byte_level_double_sequence`, `trim_offsets=true`

Use the same fixture and compare the same `ids`, `tokens`, `word_ids`,
`type_ids`, `special_tokens_mask`, and `attention_mask` as the untrimmed case.

Expected trimmed `offsets`:

`[(0, 2), (3, 7), (8, 10), (11, 18), (0, 4), (5, 7), (8, 10), (11, 15), (15, 16)]`

### `byte_level_pre_tokenized_sequence`, `trim_offsets=false`

Fixture:

- tokenizer: `get_byte_level(add_prefix_space=true, trim_offsets=false)`
- pre-tokenized input: `["My", "name", "is", "Anthonino"]`
- encode option: `add_special_tokens=false`

Expected fields:

- `ids`: `[2011, 1438, 318, 8451, 261, 2879]`
- `tokens`: `["ĠMy", "Ġname", "Ġis", "ĠAnth", "on", "ino"]`
- `offsets`: `[(0, 2), (0, 4), (0, 2), (0, 4), (4, 6), (6, 9)]`
- `word_ids`: `[0, 1, 2, 3, 3, 3]`
- `type_ids`: `[0, 0, 0, 0, 0, 0]`
- `special_tokens_mask`: `[0, 0, 0, 0, 0, 0]`
- `attention_mask`: `[1, 1, 1, 1, 1, 1]`

## Remaining Gaps

- `offsets.rs::byte_level_pre_tokenized_sequence_with_trimming` remains a
  follow-up because upstream marks it ignored.
- ByteLevel-specific pre-tokenized pair input is not claimed by this slice;
  direct and batch pre-tokenized pair API coverage is tracked in
  `padding-runtime-boundary.md`.
- Empty pre-tokenized segments, added-token extraction inside pre-tokenized
  segments, and normalized pre-tokenized inputs need separate fixtures before
  they are claimed.
- Template pair post-processing, special-token insertion, truncation, padding,
  overflowing encodings, and decode behavior remain outside this slice.
- ByteLevel `use_regex=false`, full Unicode category parity, raw non-ByteLevel
  BPE tokenization, cache/file-loader edge cases, stochastic middle-probability
  dropout fixtures, and broader decoder composition remain ByteLevel/BPE
  follow-ups. Deterministic BPE unknown fallback, byte fallback, continuing
  subword prefix, end-of-word suffix, `fuse_unk`, `ignore_merges=true`, and
  dropout `null`/`0.0`/`1.0` boundaries are covered separately in
  `bpe-config-runtime-boundary.md`.
