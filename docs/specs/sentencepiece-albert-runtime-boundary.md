# SentencePiece/ALBERT Runtime Boundary

## Source

Upstream reference data and behavior at
`22d54d37621f2d9f35cf9420d6ed8658372a6c5d`:

- `hf-internal-testing/tokenizers-test-data/albert-base-v1-tokenizer.json`
- `pre_tokenizers/whitespace.rs::whitespace_split`
- `pre_tokenizers/metaspace.rs::basic`
- `normalizers/precompiled.rs`
- `processors/template.rs` for `[CLS] $A [SEP] $B:1 [SEP]:1`
- `decoders::metaspace::Metaspace`

This is native C++ inference runtime only. It must not add Rust FFI, runtime
shell-out, trainers, training fixtures, HTTP/from-pretrained loading, or edits
under `third_party/tokenizers`.

## Accepted Behavior

The current ALBERT/SentencePiece-style path covers the deterministic components
needed by the local `albert-base-v1-tokenizer.json` smoke:

- Normalizer `Sequence` load executes the covered native subset needed by this
  fixture: `Replace` steps and ICU-backed string-level `Lowercase` with edit
  mapping.
  String and default-ICU Regex `Replace` runtime behavior is tracked more
  directly in `replace-normalizer-runtime-boundary.md`.
- `NFD`/`NFKD` execute through the vendored ICU Unicode backend. Output bytes
  are projected back to the originating input codepoint span, preserving the
  current offset model while allowing broad ICU compatibility decomposition.
- `StripAccents` removes common combining mark ranges, including Latin and Thai
  marks needed by upstream strip tests.
- `Precompiled` parses the serialized SentencePiece charsmap into the embedded
  double-array trie and normalized string, then applies the first-prefix
  transform needed by the ALBERT fixture. Covered behavior now includes
  zero-width joiner/non-joiner/space splitting, single-scalar deletion,
  chained zero-width deletion, no-break/BOM space boundaries, control separator
  mapping/deletion, compatibility expansion, and expansion followed by removal
  or joiner splitting after upstream-style NFKD, StripAccents, and lowercase.
  The charsmap lookup itself is self-contained C++ and must not call ICU, RE2,
  or other regex/Unicode engines for transform behavior.
- Pre-tokenizer `Sequence(WhitespaceSplit, Metaspace)` is executed.
- `Metaspace` supports `replacement`, legacy `add_prefix_space`, modern
  `prepend_scheme`, and `split`.
- `TemplateProcessing` supports the ALBERT/BERT shape:
  `[CLS] $A [SEP]` and `[CLS] $A [SEP] $B:1 [SEP]:1`.
- `Metaspace` decoder converts replacement markers back to spaces and drops the
  synthetic leading marker when `prepend_scheme != never`.
- Unigram inference uses the deterministic best-path runtime in
  `unigram-runtime-boundary.md`.

## Current C++ Coverage

`tests/parity/sentencepiece_albert_test.cpp` loads the real local
`albert-base-v1-tokenizer.json` and checks Rust-derived gold vectors for:

- `encode("Hello world", false)`:
  ids `[10975, 126]`, tokens `["▁hello", "▁world"]`, offsets
  `[(0, 5), (6, 11)]`, word ids `[0, 1]`, and all-zero special masks.
- `encode("Hello world", true)`:
  template insertion `[CLS] ▁hello ▁world [SEP]`, special masks, type ids,
  word ids, and `decode(..., skip_special_tokens=true) == "hello world"`.
- `encode("Héllò hôw are ü?", false)`:
  NFKD + StripAccents + Lowercase maps to tokens
  `["▁hello", "▁how", "▁are", "▁u", "?"]` while preserving byte offsets into
  the original UTF-8 input.
- `encode("ậ…", false)`:
  compatibility decomposition maps to `["▁a", ".", ".", "."]`, with each
  decomposed dot retaining the original ellipsis byte span.
- Thai strip bug shape:
  `U+0E33` compatibility decomposition plus Thai combining-mark stripping maps
  to `["▁", "านา", "3", "ลา"]`, including the standalone metaspace marker
  offset.
- `encode("A\u200dB", false)`, `encode("A\u200cB", false)`, and
  `encode("hello\u200dworld", false)`:
  the serialized `Precompiled` charsmap removes the zero-width
  joiner/non-joiner boundary and preserves Rust-derived ids, tokens, byte
  offsets, word ids, masks, and Metaspace decode output.
- `encode("hello\u001eworld", false)`:
  the serialized `Precompiled` charsmap deletes the record separator without
  introducing a new word boundary, yielding tokens `["▁hello", "world"]`,
  offsets `[(0, 5), (6, 11)]`, word ids `[0, 0]`, and decode
  `"helloworld"`.
- `encode("™\u001eg", false)`:
  NFKD expands the trademark sign, lowercase maps it to `tm`, and the
  serialized `Precompiled` charsmap deletes the record separator. The accepted
  Rust-derived output is ids `[13, 38, 11984]`, tokens `["▁", "t", "mg"]`,
  offsets `[(0, 3), (0, 3), (0, 5)]`, word ids `[0, 0, 0]`, and decode
  `"tmg"`.
- `encode("A\u200bB", false)`:
  zero-width space follows the same split behavior as the covered joiner cases,
  yielding `["▁a", "▁b"]`, offsets `[(0, 1), (4, 5)]`, word ids `[0, 1]`,
  and decode `"a b"`.
- `encode("A\u000bB\u000cC", false)`:
  the serialized `Precompiled` map deletes one control separator and converts
  the other into a word boundary, yielding `["▁ab", "▁c"]`, offsets
  `[(0, 3), (4, 5)]`, word ids `[0, 1]`, and decode `"ab c"`.
- `encode("e\u0301\u200dg", false)`:
  ICU-backed accent stripping and the serialized joiner transform compose
  before Metaspace, yielding `["▁", "e", "▁g"]`, offsets
  `[(0, 1), (0, 1), (6, 7)]`, word ids `[0, 0, 1]`, and decode `"e g"`.
- R4 hardening coverage from upstream Rust ALBERT reference output:
  - `encode("A\u00a0B", false)` and `encode("A\ufeffB", false)`:
    no-break/BOM boundaries map to `["▁a", "▁b"]` while preserving byte
    offsets around the removed/replaced separator.
  - `encode("A\u200d\u200c\u200bB", false)`:
    chained zero-width joiner/non-joiner/space deletion still yields
    `["▁a", "▁b"]`, with the second token projected to the original trailing
    `B` byte span.
  - `encode("ﬃ", false)`:
    compatibility expansion yields `["▁f", "fi"]`, both projected to the
    original ligature byte span.
  - `encode("™\u200dg", false)`:
    expansion plus joiner splitting yields `["▁", "t", "m", "▁g"]`, preserving
    the trademark span for all expanded pieces and projecting `g` to its
    original span.
- `encode_pair("Hello", "world", true)`:
  pair template ids, type ids `[0, 0, 0, 1, 1]`, offsets, word ids, and masks.
- A minimal Unigram/SentencePiece fixture with `NFKD` verifies ICU
  compatibility decompositions beyond the previous targeted table:
  `① ㍿` maps to tokens `["1", "株式会社"]` with offsets `[(0, 3), (4, 7)]`.
- A minimal Unigram/SentencePiece fixture with `Lowercase` verifies ICU
  lowercase beyond the previous targeted table:
  `Ա Բ` maps to tokens `["ա", "բ"]` with offsets `[(0, 2), (3, 5)]`.
- A minimal Unigram/SentencePiece fixture with `Lowercase` verifies
  context-sensitive Greek final sigma:
  `ΟΣ` maps to token `["ος"]` with offset `[(0, 4)]`.

The real ALBERT expected values were generated once from upstream Rust and are
committed as literal C++ assertions. The minimal ICU compatibility fixture is a
native runtime guard for behavior outside the previous targeted table. Runtime
C++ tests do not call Rust.

## Known Gaps

- Lowercase uses ICU string-level mapping with edits. Casefold remains a
  follow-up.
- The current `Precompiled` path uses conservative grapheme grouping
  approximated as one scalar plus following combining marks. It now covers the
  accepted ALBERT splitting/deletion/control/replacement/expansion fixtures,
  including chained zero-width deletion and expansion-to-multiple-token offset
  projection, but does not yet claim full `unicode-segmentation` parity across
  every possible SentencePiece charsmap.
- Metaspace behavior is covered for the ALBERT fixture shape, not every edge
  case in the upstream Metaspace unit tests.
- Direct serialized TemplateProcessing piece arrays are now handled by the
  shared post-processing path, including the broader cases recorded in
  `docs/specs/template-processing-runtime-boundary.md`. Builder/string
  template forms remain Rust construction API surfaces rather than tokenizer
  JSON accepted by this C++ port.
- SentencePiece-style `NFD`/`NFKD` now uses ICU, but exact offset behavior for
  cross-codepoint canonical reordering remains a follow-up before claiming
  broad ALBERT parity.
