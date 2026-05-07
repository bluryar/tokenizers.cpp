# Post-Processor Sequence Runtime Boundary

## Scope

This slice ports the native C++ inference path for the common serialized
post-processor `Sequence` shape used by GPT/Llama-style tokenizer JSON:
`ByteLevel` offset processing followed by `TemplateProcessing` special-token
insertion.

Covered behavior:

- tokenizer JSON `post_processor: {"type":"Sequence"}` runtime parsing
- nested `Sequence` nodes are flattened recursively for the covered child
  processors
- `ByteLevel` child config sets the native post-processing offset trimmer
- `TemplateProcessing` child config compiles explicit sequence/special-token
  pieces into the shared post-processing path
- `BertProcessing` and `RobertaProcessing` children are accepted through the
  same sequence parser
- sequence-level special-token accounting for truncation and pair processing
- special tokens referenced only by the post-processor are loaded into the C++
  vocab/special-id tables
- tokenizer JSON truncation and fixed padding compose with the covered
  `Sequence(ByteLevel, TemplateProcessing)` shape, including overflowing
  encodings
- pre-tokenized single/pair and batch/pair-batch APIs compose with truncation,
  overflowing, fixed padding, and the covered post-processor sequence while
  preserving word-relative offsets and caller-supplied word ids

Excluded behavior:

- arbitrary post-processor composition with multiple special-token processors
- order-sensitive behavior beyond the accepted `ByteLevel` before
  special-token insertion shape
- sequence range APIs, because C++ `Encoding` does not expose sequence ranges
  yet
- full Llama encode parity beyond the focused smoke fixtures in
  `llama-split-bytelevel-runtime-boundary.md`

## Accepted Fixtures

`tests/parity/post_processor_sequence_test.cpp` is the native acceptance surface
for this slice.

It builds a tokenizer from local `tokenizer.json`, then replaces the
post-processor with:

- `Sequence`
- child `ByteLevel(add_prefix_space=true, trim_offsets=true)`
- child `TemplateProcessing` using `<s>` plus `$A` / `$B`

It verifies Rust-derived single and pair ids, tokens, offsets, word ids, type
ids, special-token masks, attention masks, and ByteLevel decode behavior.

It also verifies a temporary `tokenizer.json` variant with injected truncation
and fixed padding. The fixture pins that content truncates before template
insertion, overflowing encodings receive the same `<s>` template, and fixed
padding applies recursively after post-processing.

The same temporary JSON path also verifies:

- `encode(vector<string>)` for pre-tokenized single input
- `encode_batch(vector<vector<string>>)` for ordered pre-tokenized batches
- `encode_pair(vector<string>, vector<string>)` for pre-tokenized pair input
- `encode_batch_pairs(vector<pair<vector<string>, vector<string>>>)` for
  ordered pre-tokenized pair batches

These checks pin word-relative offsets (`"a"` remains `{0, 1}` instead of the
raw-text byte span), word ids, type ids, special-token masks, attention masks,
overflow order, and recursive padding over overflowing encodings.

## Upstream References

Reference behavior comes from:

- `processors::sequence::tests::process_chain`
- `processors::template::tests::template_processing`
- a local upstream Rust probe using
  `hf-internal-testing/tokenizers-test-data/tokenizer.json` with the same
  `Sequence(ByteLevel, TemplateProcessing)` post-processor.

## Implementation Notes

The native parser converts supported `Sequence` children into the same runtime
state as direct post-processors. This keeps the encode pipeline order simple:
offset trimming still runs before the compiled special-token template, then
padding runs after post-processing as before. Temporary JSON mutation is used
only by the test harness so the upstream fixture remains an immutable reference.
