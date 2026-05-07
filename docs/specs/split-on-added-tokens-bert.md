# Split On Added Tokens BERT Boundary

## Source

Upstream target at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `third_party/tokenizers/tokenizers/tests/offsets.rs::split_on_added_tokens_bert`

This slice is native C++ inference runtime only. It must not add Rust FFI,
runtime shell-out, trainers, training fixtures, HTTP/from-pretrained loading,
wrapper APIs, or edits under `third_party/tokenizers`.

## Native C++ Boundary

Use the existing public inference API:

- `Tokenizer::from_file(path)`
- `Tokenizer::encode(text, add_special_tokens=true)`

Do not add public `add_tokens`, `add_special_tokens`, `Tokenizer::new`,
component builders, or a Rust-style mutation API for this slice. Represent the
upstream mutation
`tokenizer.add_special_tokens([AddedToken::from("[MASK]", true)])` with a local
tokenizer JSON fixture whose `added_tokens` array contains:

```json
{
  "id": 103,
  "content": "[MASK]",
  "single_word": false,
  "lstrip": false,
  "rstrip": false,
  "normalized": false,
  "special": true
}
```

Because `[MASK]` already exists in the BERT WordPiece model vocabulary, the
native loader must resolve it to model id `103`; it must not allocate a new id
from the serialized added-token record.

The fixture should model upstream `tests/common/mod.rs::get_bert()`:

- `model`: `WordPiece` loaded from the local
  `bert-base-uncased-vocab.txt` data, with `unk_token="[UNK]"`,
  `continuing_subword_prefix="##"`, and
  `max_input_chars_per_word=100`.
- `normalizer`: default `BertNormalizer` JSON:
  `clean_text=true`, `handle_chinese_chars=true`, `strip_accents=null`,
  `lowercase=true`.
- `pre_tokenizer`: `{"type":"BertPreTokenizer"}`.
- `decoder`: default WordPiece decoder JSON, if present:
  `{"type":"WordPiece","prefix":"##","cleanup":true}`.
- `post_processor`: `BertProcessing` with `sep=["[SEP]",102]` and
  `cls=["[CLS]",101]`.

## Runtime Semantics

The native encode path for this slice must follow the upstream order:

1. Extract added tokens before normal pre-tokenization and model tokenization.
   The `[MASK]` record has `normalized=false`, so it matches the original
   uppercase input before `BertNormalizer` lowercases ordinary text spans.
2. Apply `BertNormalizer` to non-added spans. The exact upstream test only
   observes default ASCII lowercasing and byte-offset preservation, but the
   component state should keep the upstream fields so later Unicode, accent, and
   Chinese-character fixtures can extend behavior without changing the JSON
   boundary.
3. Apply `BertPreTokenizer` to non-added spans. For this test it must remove
   whitespace delimiters while preserving original byte offsets. Punctuation
   isolation is part of the component's upstream behavior, but this test does
   not exercise it because `[MASK]` is an added-token split.
4. Tokenize normal pre-tokenized pieces with WordPiece greedy longest matching
   on UTF-8 character boundaries, using the configured `##` continuing-subword
   prefix, `[UNK]` fallback, and `max_input_chars_per_word`. The exact test
   words are whole-vocabulary hits; do not replace this with WordLevel lookup
   behavior.
5. Keep the added `[MASK]` split as a direct encoded token with id `103`,
   token string `[MASK]`, and original byte offset `(18, 24)`.
6. Run `BertProcessing` only through its `add_special_tokens=false` branch. It
   must return the encoding unchanged: no `[CLS]`, no `[SEP]`, no `(0, 0)`
   special offsets, and no special-token mask entries from post-processing.

The WordPiece decoder is not used by the upstream test because it never calls
`decode`. This slice may parse and store the decoder for full `get_bert()`
JSON compatibility, but it must not claim decode parity until a decoder test is
added.

## Exact Acceptance

Fixture:

- tokenizer: upstream-equivalent `get_bert()` plus special added token
  `[MASK]`.
- input: `"Yesterday I saw a [MASK] far away"`
- encode option: `add_special_tokens=false`

Expected fields:

- `ids`: `[7483, 1045, 2387, 1037, 103, 2521, 2185]`
- `tokens`:
  `["yesterday", "i", "saw", "a", "[MASK]", "far", "away"]`
- `offsets`:
  `[(0, 9), (10, 11), (12, 15), (16, 17), (18, 24), (25, 28), (29, 33)]`
- `word_ids`: `[0, 1, 2, 3, 4, 5, 6]`
- `type_ids`: `[0, 0, 0, 0, 0, 0, 0]`
- `special_tokens_mask`: `[0, 0, 0, 0, 0, 0, 0]`
- `attention_mask`: `[1, 1, 1, 1, 1, 1, 1]`

The `[MASK]` token is special for added-vocabulary registration and decode
skipping, but upstream `Encoding::special_tokens_mask` remains all zero here
because no post-processor special tokens are inserted when
`add_special_tokens=false`.

## Remaining Gaps

- Public tokenizer mutation APIs remain out of scope.
- `BertProcessing` sequence ranges, truncation interaction, padding, and
  overflowing encodings remain follow-ups. Single and pair insertion are covered
  separately in `wordpiece-bert-processing-breadth.md`.
- WordPiece decoder behavior, `BertNormalizer` Unicode cleanup/accent handling,
  and `BertPreTokenizer` punctuation coverage are now covered by follow-on
  fixtures in `wordpiece-bert-processing-breadth.md`.
- The exact upstream vector only exercises whole-token WordPiece hits. Greedy
  subword, unknown-token, and max-input-character coverage are tracked
  separately in `wordpiece-bert-processing-breadth.md`.

## Verification Commands

- Upstream exact reference:
  `cd projects/tokenizers.cpp/third_party/tokenizers/tokenizers && CARGO_TARGET_DIR=<repo>/build/rust-target cargo test --test offsets split_on_added_tokens_bert -- --exact`
- C++ parity target:
  `ctest --test-dir projects/tokenizers.cpp/build -R tokenizers_cpp_bert_wordpiece_added_tokens_test --output-on-failure`
- Full C++ parity suite:
  `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`
