# Roberta Processing Runtime Boundary

## Scope

This slice ports the native C++ inference path for direct
`RobertaProcessing` post-processors and broadens serialized
`TemplateProcessing` handling from the previous BERT/ALBERT-shaped shortcut
into explicit template pieces.

Covered behavior:

- tokenizer JSON `RobertaProcessing` load with:
  - `sep`
  - `cls`
  - `trim_offsets`
  - `add_prefix_space`
- single sequence post-processing:
  - `<s> $A </s>`
- pair sequence post-processing:
  - `<s> $A </s> </s> $B </s>`
- RoBERTa all-zero `type_ids`, including `add_special_tokens=false`
- ByteLevel offset trimming through the RoBERTa processor settings
- truncation accounting for RoBERTa pair processing with four special tokens
- tokenizer JSON truncation and fixed padding over real RoBERTa single-sequence
  encodings, including post-processed overflowing encodings
- pre-tokenized single/pair and batch/pair-batch overloads over the temporary
  real RoBERTa truncation/padding JSON, preserving word-relative offsets and
  word ids through overflowing encodings
- direct serialized `TemplateProcessing` piece arrays:
  - `Sequence` pieces for `A` and `B`
  - `SpecialToken` references through `special_tokens`
  - per-piece `type_id`
  - multi-id/multi-token special-token records
- post-processor `Sequence` runtime composition for the covered
  `ByteLevel + TemplateProcessing` shape is handled in
  `post-processor-sequence-runtime-boundary.md`

Excluded behavior:

- string-builder TemplateProcessing forms; the accepted runtime path is the
  serialized JSON piece-array form
- sequence range APIs, because C++ `Encoding` does not expose sequence ranges
  yet
- arbitrary multi-special-processor composition inside post-processor
  `Sequence`
- full RoBERTa tokenizer families outside the local `roberta.json` smoke and
  existing ByteLevel/BPE coverage

## Accepted Fixtures

`tests/parity/roberta_processing_test.cpp` is the native acceptance surface for
this slice.

It verifies:

- real local `roberta.json` single encode with and without special tokens
- real local `roberta.json` pair encode with and without special tokens
- exact ids, tokens, offsets, word ids, type ids, special-token masks, and
  attention masks from an upstream Rust reference probe
- ByteLevel decode with RoBERTa special tokens kept and skipped
- real local `roberta.json` with injected `truncation` and `padding` settings:
  - content is truncated before RoBERTa special-token insertion
  - main and overflowing encodings both receive `<s>` / `</s>`
  - fixed padding is applied recursively after post-processing
  - ids, tokens, offsets, word ids, type ids, special-token masks, and
    attention masks are pinned for both main and overflowing encodings
  - pre-tokenized single and batch inputs keep offsets relative to each input
    word while producing RoBERTa tokens without raw-text leading-space markers
  - pre-tokenized pair and pair-batch inputs use a larger temporary max length
    so RoBERTa's four pair specials and two content tokens fit, then assert
    pair overflowing encodings and all-zero RoBERTa type ids
- a small WordLevel + `RobertaProcessing` tokenizer where pair truncation
  proves that four RoBERTa special tokens are counted before post-processing

## Upstream References

Reference behavior comes from:

- `processors::roberta::tests::serde`
- `processors::roberta::tests::roberta_processing`
- `processors::template::tests::template_processing`
- the local upstream Rust tokenizer loaded from
  `hf-internal-testing/tokenizers-test-data/roberta.json`

## Implementation Notes

The runtime now compiles BERT, RoBERTa, and direct serialized
`TemplateProcessing` into explicit post-processing pieces. Each piece is either
sequence A, sequence B, or a special token span. Truncation accounting sums the
special-token spans in the selected single or pair template instead of assuming
the old BERT-only `2/3` shape.

Real-tokenizer smoke tests inject truncation and padding into a temporary copy
of the upstream JSON fixture. The committed HF fixture remains read-only; the
temporary JSON only exists long enough to exercise the native load path.
