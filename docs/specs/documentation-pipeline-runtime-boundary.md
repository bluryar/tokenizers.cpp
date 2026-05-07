# Documentation Pipeline Runtime Boundary

## Source

This slice translates the inference-only parts of
`third_party/tokenizers/tokenizers/tests/documentation.rs::pipeline`.

Training, Rust builder APIs, public typed component builders, and serializer
roundtrips remain out of scope. The C++ test uses an equivalent temporary
tokenizer JSON loaded through `Tokenizer::from_file`.

## Covered Behavior

- Real local `tokenizer-wiki.json` BPE model and added special tokens.
- Serialized `Sequence(NFD, StripAccents)` normalizer before BPE.
- Serialized `Sequence(Whitespace, Digits(individual_digits=true))`
  pre-tokenizer before non-ByteLevel BPE.
- Serialized `TemplateProcessing` for `[CLS] $A [SEP]` and
  `[CLS] $A [SEP] $B:1 [SEP]:1`.
- Tokenizer-level `encode(..., add_special_tokens=true)` fields:
  ids, tokens, byte offsets, word ids, type ids, special-token mask, and
  attention mask.
- `decode(ids, skip_special_tokens=true)` over the same ids, including
  `[UNK]` special-token skipping.

## Accepted Fixture

Input:

`Hello, y'all! How are you 😁 ?`

Expected tokens:

`[CLS] Hello , y ' all ! How are you [UNK] ? [SEP]`

Expected ids:

`[1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]`

Expected decoded text with special tokens skipped:

`Hello , y ' all ! How are you ?`

## Known Gaps

- This does not add public C++ `with_normalizer`, `with_pre_tokenizer`, or
  `with_post_processor` builder-style mutation APIs. Runtime assembly is
  represented by tokenizer JSON loading, which is the accepted public surface
  for the native inference port.
- Broad arbitrary pre-tokenizer sequence composition remains incremental. This
  slice covers the `Whitespace -> Digits` order needed by the documentation
  pipeline and keeps existing `Split -> ByteLevel` behavior intact.
