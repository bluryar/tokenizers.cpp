# Llama Split ByteLevel Runtime Boundary

## Scope

This slice ports the native C++ inference path for the Llama/GPT-style
pre-tokenizer sequence:

- `Split` with the serialized Llama regex
- `behavior = "Isolated"`
- `invert = false`
- `ByteLevel(add_prefix_space=false, use_regex=false)`

Together with the previously covered post-processor `Sequence`, this lets the
real local `llama-3-tokenizer.json` run encode-side smoke tests instead of only
decode-side tests.

Covered behavior:

- tokenizer JSON `pre_tokenizer: {"type":"Sequence"}` runtime parsing for the
  covered `Split + ByteLevel` shape
- Llama-style regex splitting for:
  - case-insensitive ASCII contractions
  - letter spans with optional leading non-letter/non-number prefix
  - numeric spans of one to three digits
  - Unicode letter and number classification through the project Unicode
    backend; the default vendored ICU backend covers broad `\p{L}` / `\p{N}`
  - punctuation/symbol spans
  - whitespace spans, including CR/LF boundaries and multi-space runs before
    a following word
- `ByteLevel(use_regex=false)` after `Split`, preserving each split piece as a
  single byte-level input piece
- real local Llama BPE encode with template special-token insertion
- tokenizer JSON truncation and fixed padding over the real local Llama path,
  including overflowing encodings
- public `encode_batch` and `encode_batch_pairs` over the same temporary
  truncation/padding settings, preserving input order and padding both main and
  overflowing encodings
- public pre-tokenized `encode_batch` and `encode_batch_pairs` over the same
  temporary truncation/padding settings, preserving word-relative offsets and
  caller-supplied word ids
- public `encode_batch_char_offsets` and `encode_batch_pairs_char_offsets`
  over the same temporary truncation/padding settings

Excluded behavior:

- Broad non-Llama `Split` behavior is tracked in
  `split-pretokenizer-runtime-boundary.md`; this Llama slice still only claims
  the serialized `Split + ByteLevel(use_regex=false)` tokenizer shape.
- full Rust-regex parity for arbitrary `Split` patterns beyond the accepted ICU
  backend fixtures
- full multilingual Llama encode parity beyond the accepted Latin/CJK/fullwidth
  digit and ICU-only category fixture vectors
- runtime ByteLevel normalizer mutation APIs

## Accepted Fixtures

`tests/parity/llama_encode_test.cpp` is the native acceptance surface for this
slice.

It loads the real local `llama-3-tokenizer.json` and verifies Rust-derived:

- `encode("Hey! how is this token: ", false)`
- `encode("Hello, world!", true)`
- `encode("abc 1234", false)`, including three-digit number grouping
- `encode("I'm you're they'll don't", false)`, including contraction splits
- `encode("Hello\nworld\r\n  next", false)`, including newline and
  multi-space splitting
- `encode("café 東京 １２3", false)`, including Latin-extended/CJK letters and
  fullwidth digit classification
- `encode("AሴB Ⅻ3", false)` in the default vendored ICU build, including
  Ethiopic letter and Roman numeral number classification
- `encode_pair("Hello", "world", true)`
- a temporary `llama-3-tokenizer.json` variant with injected truncation and
  fixed padding:
  - `encode("Hello, world!", true)` truncates content before template insertion
  - main and overflowing encodings both receive `<|begin_of_text|>`
  - `<|finetune_right_pad_id|>` pads both encodings to the fixed length
- `encode_batch({"Hello, world!", "Hello"}, true)` over the same temporary
  JSON, checking ordered outputs and fixed padding across a short item and an
  overflowing item
- `encode_batch({{"Hello,", "world!"}, {"Hello"}}, true)` over the same
  temporary JSON, checking pre-tokenized word-relative offsets, word ids,
  truncation, overflowing, and fixed padding
- `encode_pair("Hello, world!", "world", true)` over the same temporary JSON,
  checking LongestFirst content allocation after the two Llama pair-template
  specials are counted, plus three overflowing pair encodings
- `encode_batch_pairs({{"Hello, world!", "world"}, {"Hello", "test"}}, true)`
  over the same temporary JSON, checking ordered pair outputs, type ids, and
  fixed padding across an overflowing pair and a short pair
- pre-tokenized `encode_batch_pairs({{{"Hello,", "world!"}, {"world"}},
  {{"Hello"}, {"test"}}}, true)` over the same temporary JSON, checking
  word-relative pair offsets and word ids across main and overflowing encodings
- `encode_batch_char_offsets({"café 東京 １２3", "Hello"}, true)` over the
  same temporary JSON, checking scalar char offsets across Unicode content,
  overflowing encodings, template specials, and padding
- `encode_batch_pairs_char_offsets({{"Hello, world!", "world"}, {"Hello",
  "test"}}, true)` over the same temporary JSON, checking pair-batch
  char-offset API composition after truncation and padding

Each fixture checks ids, tokens, offsets, word ids, type ids, special-token
masks, attention masks, and selected ByteLevel decode behavior.

## Upstream References

Reference behavior comes from:

- `pre_tokenizers::split::tests::basic`
- `pre_tokenizers::byte_level::tests::pre_tokenization_no_regex`
- `processors::sequence::tests::process_chain`
- a local upstream Rust probe using
  `hf-internal-testing/tokenizers-test-data/llama-3-tokenizer.json`

## Implementation Notes

The Llama regex is recognized by shape and lowered to explicit UTF-8 scanning
logic. Letter and number classification now call the shared Unicode backend:
vendored static ICU4C in the default build. The runtime still does not use RE2,
PCRE2, or a system regex/Unicode dynamic library for this path.

The truncation/padding fixture mutates a temporary copy of the local tokenizer
JSON. This keeps the HF fixture read-only while still exercising the same
native `Tokenizer::from_file` load path as downstream consumers.
