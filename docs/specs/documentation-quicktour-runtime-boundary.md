# Documentation Quicktour Runtime Boundary

## Source

This slice translates the inference-only assertions from
`third_party/tokenizers/tokenizers/tests/documentation.rs::quicktour`.

Training, saving, and Rust builder-style mutation APIs remain out of scope.
The C++ tests express post-processor and padding changes as temporary tokenizer
JSON loaded through `Tokenizer::from_file`.

## Covered Behavior

- Real local `tokenizer-wiki.json` load.
- Direct `Whitespace` pre-tokenizer plus BPE model encode for:
  `Hello, y'all! How are you 😁 ?`
- Initial quicktour fields before any post-processor:
  tokens, ids, byte offsets, word ids, type ids, special-token mask,
  attention mask, and `token_to_id("[SEP]") == 2`.
- TemplateProcessing quicktour state:
  `[CLS] $A [SEP]` and `[CLS] $A [SEP] $B:1 [SEP]:1`.
- Single and pair encode after TemplateProcessing, including pair `type_ids`.
- Ordered `encode_batch` and `encode_batch_pairs` execution on the quicktour
  inputs.
- BatchLongest padding with `pad_id=3`, `pad_token="[PAD]"`, right padding,
  and quicktour attention-mask behavior.

## Accepted Fixtures

Initial encode:

- tokens:
  `Hello , y ' all ! How are you [UNK] ?`
- ids:
  `[27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]`
- emoji offset:
  `(26, 30)`

Template pair:

- tokens:
  `[CLS] Hello , y ' all ! [SEP] How are you [UNK] ? [SEP]`
- type ids:
  `[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]`

Batch padding second item:

- tokens:
  `[CLS] How are you [UNK] ? [SEP] [PAD]`
- attention mask:
  `[1, 1, 1, 1, 1, 1, 1, 0]`

## Known Gaps

- This does not add public C++ `with_post_processor` or `with_padding`
  mutation APIs. The accepted public assembly path remains tokenizer JSON load.
- The upstream quicktour's batch-pair example only prints its second item
  rather than asserting exact fields. The C++ test therefore asserts exact
  fields for the first batch pair, and structural execution for the second.
