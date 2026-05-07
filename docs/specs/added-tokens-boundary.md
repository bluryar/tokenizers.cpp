# Added Tokens Boundary

## Source

Semantic reference:
`third_party/tokenizers/tokenizers/tests/added_tokens.rs` at
`22d54d37621f2d9f35cf9420d6ed8658372a6c5d`.

This slice translates added-token runtime behavior into native C++ inference
semantics. The upstream Rust tests use mutable `add_tokens` and
`add_special_tokens` helpers, but the C++ public surface remains the existing
inference API unless a later ADR expands it:

- `Tokenizer::from_file(path)`
- `Tokenizer::encode(text, add_special_tokens=true)`
- `Tokenizer::encode_pair(text_a, text_b, add_special_tokens=true)`
- `Tokenizer::decode(ids, skip_special_tokens=true)`
- `Tokenizer::token_to_id(token)`
- `Tokenizer::id_to_token(id)`

Use local tokenizer JSON fixtures or internal construction in C++ tests to
represent the same runtime state. Do not add Rust FFI, runtime shell-out,
training, HTTP/from-pretrained loading, wrapper APIs, or edits under
`third_party/tokenizers`.

## Native Runtime Boundary

Implement an internal added-token runtime object with these upstream fields:

- `id`: unsigned 32-bit token id assigned by the runtime added-vocabulary
  registration path.
- `content`: exact token text.
- `single_word`: require word-boundary matching around the token.
- `lstrip`: include contiguous whitespace to the left in the emitted token span.
- `rstrip`: include contiguous whitespace to the right in the emitted token span.
- `normalized`: match against normalizer output when true, otherwise match the
  original string.
- `special`: mark the token for decode skipping and special added-vocabulary
  handling. Upstream does not set `Encoding::special_tokens_mask` for added
  tokens encountered in the input; that mask is reserved for post-processor
  inserted tokens.

`added_tokens` may be absent or empty. A present record must be an object with
valid field types; malformed records are load errors, not skipped entries. The
serialized `id` field is required because upstream JSON contains it, but
deserialization must follow upstream `Tokenizer::from_file`: warn-or-ignore id
mismatches and route records through added-token registration, where existing
model vocabulary ids win and new ids are allocated after the current
vocabulary. The loader should preserve Hugging Face field names and store
added-token metadata separately from the model vocabulary even when lookup
tables are merged for `token_to_id`.

The next slice should not add a public C++ `add_tokens` or `add_special_tokens`
mutation API. If future work needs one, it should be explicit API design, not a
side effect of porting these tests.

## Encode Semantics

Added-token extraction runs before normal pre-tokenization and model
tokenization. `add_special_tokens=false` disables post-processor insertion of
extra special tokens; it does not disable matching added tokens already present
in the input.

Matching rules to port from `added_tokens.rs`:

- ID assignment and lookup: added special tokens and normal added tokens must be
  visible through `token_to_id`; normal added tokens follow the model vocab and
  earlier added ids when ids are allocated internally.
- Strip flags: `lstrip=true` extends the added-token span left across adjacent
  whitespace; `rstrip=true` extends it right across adjacent whitespace. The
  token text and byte offsets must include the consumed whitespace.
- Prefix-space interaction: `rstrip=true` must preserve the upstream
  byte-level behavior where an added trailing space and a byte-level
  `add_prefix_space=true` pre-tokenizer can still cause the following model
  token to carry a prefix-space marker.
- `single_word=true`: reject matches embedded inside word characters, such as
  matching `ing` inside `dancing`. `single_word=false` may split such words.
- Overlaps: choose leftmost-longest matches. Insertion order is not part of the
  accepted behavior for overlapping tokens.
- Offsets: output offsets are byte offsets into the original input. Added
  tokens that consume whitespace through `lstrip` or `rstrip` must report the
  expanded original span.

The extractor must produce normal pieces and added-token pieces that cover the
input in order, then pass only normal pieces through the configured
pre-tokenizer/model path. Added-token pieces become encoded tokens directly
with their added-token ids, type ids, offsets, attention mask entries, word ids,
and all-zero special-token mask entries unless a later post-processor inserts
tokens.

## Acceptance Mapping

Port these upstream cases as native C++ parity coverage when implementation
starts:

- `add_tokens`: prove added special-token and normal-token lookup ids.
- `single_word_tokens`: prove whole-word gating versus suffix splitting.
- `overlapping_tokens`: prove leftmost-longest precedence and no insertion-order
  dependency.
- `lstrip_tokens`: the current WordLevel path proves leading-whitespace span
  capture only. Full upstream parity belongs to
  `byte-level-bpe-runtime-boundary.md` and requires ByteLevel pre-tokenization,
  deterministic GPT-2 BPE, ByteLevel offset processing, and Unicode byte
  offsets.
- `rstrip_tokens`: the current WordLevel path proves trailing-whitespace span
  capture only. Full upstream parity belongs to
  `byte-level-bpe-runtime-boundary.md` and requires ByteLevel prefix-space
  interaction after added-token extraction.

Use committed fixtures generated from the upstream Rust reference when expected
ids, tokens, or offsets are non-trivial. Generated fixture production may use
Rust at development time, but C++ runtime tests must not shell out to Rust.

## Out Of Scope

- Public mutable tokenizer-authoring APIs.
- Training, trainers, sampling, HTTP/from-pretrained loading, benchmarks, or
  Python/Node/WASM wrappers.
- C++ JSON serialization output parity.
- Broad offset-suite coverage outside the added-token cases above.
- Patching or vendoring changes into `third_party/tokenizers`.
