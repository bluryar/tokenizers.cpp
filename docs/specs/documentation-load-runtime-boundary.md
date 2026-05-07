# Documentation Load Runtime Boundary

## Source

Semantic reference:

- `third_party/tokenizers/tokenizers/tests/documentation.rs::load_tokenizer`
- `hf-internal-testing/tokenizers-test-data/roberta.json`

This slice remains native C++ inference only. It must not add Rust FFI, runtime
shell-out, trainers, HTTP/from-pretrained loading, or edits under
`third_party/tokenizers`.

## Covered Behavior

The C++ parity test loads the real local `roberta.json` fixture with
`Tokenizer::from_file`, matching the upstream documentation example:

- `encode("This is an example", false)` ids are `{713, 16, 41, 1246}`.
- Encoded tokens are `This`, `Ġis`, `Ġan`, and `Ġexample`.
- `decode({713, 16, 41, 1246}, false)` returns `This is an example`.

## Known Gaps

- This spec only claims the documentation load example. Broader RoBERTa
  processing, truncation, padding, overflowing, char offsets, and
  pre-tokenized batch coverage live in `roberta_processing_test.cpp` and its
  related runtime specs.
