# Byte-Level BPE Runtime Boundary

## Source

Upstream targets at `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `third_party/tokenizers/tokenizers/tests/added_tokens.rs::lstrip_tokens`
- `third_party/tokenizers/tokenizers/tests/added_tokens.rs::rstrip_tokens`
- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_basic`
- `third_party/tokenizers/tokenizers/tests/offsets.rs::byte_level_unicode`

This slice is native C++ inference runtime only. It must not add Rust FFI,
runtime shell-out, trainers, training fixtures, HTTP/from-pretrained loading,
wrapper APIs, or edits under `third_party/tokenizers`.

## Smallest Runtime Boundary

Keep the public C++ API unchanged:

- `Tokenizer::from_file(path)`
- `Tokenizer::encode(text, add_special_tokens=true)`
- `Tokenizer::decode(ids, skip_special_tokens=true)`
- `token_to_id`, `id_to_token`, and `get_vocab_size`

Represent the upstream helper `get_byte_level(add_prefix_space, trim_offsets)`
with a local tokenizer JSON/runtime graph:

- `model`: deterministic `BPE` loaded from local GPT-2 `vocab` and ordered
  `merges`.
- `pre_tokenizer`: `ByteLevel` with `add_prefix_space` and `use_regex=true`.
  A later accepted fixture also covers `Sequence(Split, ByteLevel)` with a
  generic ICU regex `Split` child and `ByteLevel(use_regex=false)`.
- `post_processor`: `ByteLevel` with `trim_offsets`. For the accepted
  `trim_offsets=true` fixture this also has `add_prefix_space=true`, matching
  upstream `ByteLevel::default().trim_offsets(true)`.
- `decoder`: `ByteLevel`.
- `normalizer`: absent or `null` for the original byte-level slice; the
  accepted regex-composition fixture also covers a serialized `Replace`
  normalizer before ByteLevel/BPE.
- `added_tokens`: JSON records for `<mask>` in the added-token tests.

The implementation may use internal component structs, but this slice should
not introduce public component construction or mutation APIs.

## Byte-Level Pre-Tokenizer

Implement enough upstream `ByteLevel` behavior to cover the four source tests:

- Preserve the GPT-2 byte-to-unicode alphabet exactly, including the space
  marker `Ġ`.
- When `add_prefix_space=true`, prepend one synthetic space to each normal
  split that does not already start with a space. Synthetic spaces affect the
  byte-level token string but do not add original-input bytes to offsets.
- When `use_regex=true`, split with the upstream GPT-2 regex semantics:
  contractions, optional-leading-space letter runs, optional-leading-space
  number runs, optional-leading-space non-space/non-letter/non-number runs,
  trailing whitespace, and remaining whitespace. The accepted fixtures exercise
  ASCII words/numbers plus non-ASCII symbol bytes such as `⭢` and `😺`; full
  Unicode letter/number category parity remains a follow-up.
- Map every UTF-8 byte of each split through the byte-to-unicode table before
  BPE. Multi-byte Unicode scalars become multiple byte-level characters that
  can later produce overlapping original byte offsets.
- Track offset alignment back to byte offsets in the original input, not UTF-16,
  codepoint, or normalized-string indexes.

`use_regex=false` is covered for the accepted
`Sequence(Split, ByteLevel(use_regex=false))` composition fixture. Broader
standalone `use_regex=false`, pre-tokenized input arrays, ByteLevel normalizer
behavior, and full Unicode normalization remain outside this slice.

## BPE Model

Implement deterministic BPE inference for the loaded GPT-2 fixture:

- Store `vocab` as token-to-id and id-to-token lookup tables.
- Parse ordered `merges` into pair ranks. Accept both serialized forms already
  used by the loader, string entries such as `"a b"` and two-string arrays.
- Validate that both merge inputs and the concatenated merge output exist in
  the vocabulary. Invalid merge references are load errors.
- Encode a byte-level pre-tokenized segment by starting from byte-level
  characters, applying the lowest-rank merge first, and using leftmost position
  as the tie-breaker for equal rank.
- Produce model tokens with ids, token strings, and offsets relative to the
  byte-level segment, then project those offsets back to original input byte
  spans through the pre-tokenizer alignment.
- Preserve overlapping offsets when multiple BPE tokens cover different bytes
  of the same original Unicode scalar. This is required for `😺` and `⭢`.

For this slice, accepted BPE configuration is the deterministic GPT-2 shape:
`dropout` absent, `null`, or `0`; `unk_token=null`;
`continuing_subword_prefix=null`; `end_of_word_suffix=null`;
`fuse_unk=false`; `byte_fallback=false`; and `ignore_merges=false`.
Additional deterministic BPE configuration behavior is specified separately in
`bpe-config-runtime-boundary.md`.

## Added-Token Interaction

Added-token extraction still runs before normal pre-tokenization and BPE.
For this byte-level slice:

- Use leftmost-longest matching from the existing added-token boundary.
- `lstrip=true` expands the added-token span left across contiguous whitespace;
  the emitted added token text and offset include that whitespace.
- `rstrip=true` expands the span right across contiguous whitespace; the emitted
  added token text and offset include that whitespace.
- Normal text before and after added-token spans is pre-tokenized separately.
  Therefore, with `add_prefix_space=true`, a normal segment after an
  `rstrip=true` added token still receives a synthetic prefix space. This is the
  upstream behavior behind `["<mask> ", "ĠðŁĺ", "º"]`.
- Added tokens bypass BPE and become encoding entries directly, but they still
  participate in ids, type ids, offsets, word ids, masks, attention masks, and
  decode special-token skipping.

## Byte-Level Post-Processor

Implement ByteLevel offset processing only to the extent needed here:

- If `trim_offsets=false`, preserve offsets produced by added-token extraction,
  byte-level alignment, and BPE.
- If `trim_offsets=true`, trim leading and trailing real whitespace represented
  by either ordinary whitespace or byte-level space marker `Ġ`.
- Do not trim the single synthetic prefix space from the first token when
  `add_prefix_space=true`.
- Preserve sequence ids/type ids for single-sequence encodings as upstream does.
  Pair offset handling for the next slice is defined in
  `byte-level-pair-pretokenized-offsets.md`.

## Byte-Level Decoder

Implement the `ByteLevel` decoder as the inverse of the byte-to-unicode mapping:

- Resolve ids through added vocabulary plus model vocabulary.
- Honor `skip_special_tokens`.
- Convert mapped byte-level characters back to bytes and decode the resulting
  UTF-8 lossily in the same spirit as upstream `String::from_utf8_lossy`.
- If a token contains characters outside the byte-level alphabet, preserve the
  token bytes rather than dropping the token.

Streaming decode and decoder `Sequence` composition are outside this slice.

## Exact Acceptance

Use committed C++ fixtures generated from the upstream Rust reference at the
source commit above. Generation may use Rust at development time, but C++
runtime tests must load local JSON/data and must not shell out.

Required direct assertions:

- `offsets.rs::byte_level_basic` with
  `input = "Hello there, how are you?"`, `add_prefix_space=true`,
  `trim_offsets=false`:
  offsets must slice the original input as
  `["Hello", " there", ",", " how", " are", " you", "?"]`, i.e.
  `[(0,5),(5,11),(11,12),(12,16),(16,20),(20,24),(24,25)]`.
- `offsets.rs::byte_level_basic` with the same input and
  `trim_offsets=true`:
  offsets must slice as
  `["Hello", "there", ",", "how", "are", "you", "?"]`, i.e.
  `[(0,5),(6,11),(11,12),(13,16),(17,20),(21,24),(24,25)]`.
- `offsets.rs::byte_level_unicode` with `input = "i⭢j"`,
  `add_prefix_space=true`, `trim_offsets=false`:
  output offsets at token indexes `1`, `2`, and `3` must all slice the original
  input to `"⭢"`, i.e. each span is `(1,4)`.
- `added_tokens.rs::lstrip_tokens` with
  `input = "I saw a <mask> 😺"`, `add_prefix_space=true`,
  `trim_offsets=false`, and special added token
  `{content:"<mask>", lstrip:true}`:
  tokens must be
  `["ĠI","Ġsaw","Ġa"," <mask>","ĠðŁĺ","º"]`
  and offsets must be
  `[(0,1),(1,5),(5,7),(7,14),(14,19),(15,19)]`.
- `added_tokens.rs::rstrip_tokens` with the same input,
  `add_prefix_space=false`, `trim_offsets=false`, and special added token
  `{content:"<mask>", rstrip:true}`:
  tokens must be
  `["I","Ġsaw","Ġa","Ġ","<mask> ","ðŁĺ","º"]`.
- `added_tokens.rs::rstrip_tokens` with `add_prefix_space=true`:
  tokens must be
  `["ĠI","Ġsaw","Ġa","Ġ","<mask> ","ĠðŁĺ","º"]`.
- ICU `Replace` regex before ByteLevel/BPE:
  with `normalizer = Replace(Regex("\\p{P}+") -> " ")`, direct
  `ByteLevel(add_prefix_space=false, use_regex=true)`, and
  `input = "hello𐄀world"`, tokens must be `["hello", "Ġworld"]`, offsets
  `[(0,5),(5,14)]`, word ids `[0,1]`, and decode `"hello world"`.
- ICU `Split` regex before ByteLevel/BPE:
  with `pre_tokenizer = Sequence(Split(Regex("\\p{P}+"), Isolated),
  ByteLevel(add_prefix_space=false, use_regex=false))` and the same input,
  tokens must be `["hello","ð","Ĳ","Ħ","Ģ","world"]`, offsets
  `[(0,5),(5,9),(5,9),(5,9),(5,9),(9,14)]`, word ids
  `[0,1,1,1,1,2]`, and decode `"hello𐄀world"`.

C++ parity fixtures for these cases should also compare ids, type ids, word
ids, special-token masks, attention masks, and decoded text whenever the
fixture records them. Do not weaken offset comparisons to substring presence
except where the upstream Rust source itself only asserts substrings.

## Known Gaps After This Slice

- `offsets.rs::{byte_level_double_sequence,byte_level_pre_tokenized_sequence}`
  are accepted in `byte-level-pair-pretokenized-offsets.md`.
- `offsets.rs::byte_level_pre_tokenized_sequence_with_trimming`, which upstream
  marks ignored.
- `offsets.rs::split_on_added_tokens_bert`, which is now scoped separately in
  `split-on-added-tokens-bert.md`.
- Broad ByteLevel `use_regex=false` fixtures beyond the accepted
  `Split + ByteLevel` regex-composition smoke.
- Full Unicode letter/number category parity in the ByteLevel regex splitter.
- ByteLevel normalizer and broad Unicode normalization/case behavior beyond
  the accepted `Replace` regex composition smoke.
- BPE dropout and raw non-ByteLevel BPE tokenization.
- Unigram inference parity.
- Post-processor templates, truncation, padding, overflowing encodings, and
  pair sequence post-processing.
- Streaming decode and decoder sequence composition.
