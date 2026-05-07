# WordLevel Runtime Boundary

## Source

Semantic reference:
`third_party/tokenizers/tokenizers/src/models/wordlevel/mod.rs`.

This slice ports only native C++ inference behavior for the WordLevel model.
It does not add training, Rust FFI, runtime shell-out, HTTP/from-pretrained
loading, wrappers, or edits under `third_party/tokenizers`.

## Covered Behavior

The C++ runtime now covers:

- WhitespaceSplit-style WordLevel encode on original byte offsets.
- Direct vocab hit: emitted token text is the input piece and offsets cover the
  piece.
- Unknown fallback: when a piece is absent from `vocab`, emit configured
  `unk_token` with the unk id while preserving the original piece offset.
- Missing-unk failure: known tokens still encode, but an unknown input throws if
  the configured `unk_token` is not present in the vocab.
- Encoding vectors remain aligned: ids, tokens, offsets, word ids, type ids,
  special-token mask, and attention mask.

Coverage lives in `tests/parity/wordlevel_test.cpp`.

## Remaining Gaps

- Non-whitespace pre-tokenizer combinations for WordLevel remain broader
  pipeline work.
- WordLevel decode remains covered only by generic token-id lookup/joining, not
  a dedicated upstream fixture.
- Truncation, padding, overflowing encodings, and post-processor interactions
  remain follow-ups.
