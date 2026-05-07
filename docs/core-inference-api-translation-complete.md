# Core Inference API Translation Complete

Date: 2026-05-07

## Verdict

The tokenizer-centered Rust-to-C++ inference translation is milestone-complete.

`tokenizers.cpp` now provides a native C++ inference runtime for the accepted
public API surface without Rust FFI, runtime Rust shell-out, Python wrappers, or
network loading. The project can load local Hugging Face tokenizer JSON files
and run the main inference workflows used by downstream C++ model runtimes.

This is not a claim of full Hugging Face `tokenizers` crate parity. It is a
scoped completion statement for native C++ inference.

## Public Inference Surface

The completed tokenizer-centered surface includes:

- `Tokenizer::from_file(path)`
- `Tokenizer::from_bpe_files(vocab_path, merges_path, options={})`
- single encode over raw strings, pre-tokenized words, and char-offset mode
- pair encode over raw strings, pre-tokenized words, and char-offset mode
- ordered batch encode for single and pair inputs
- ordered batch char-offset encode for single and pair inputs
- `decode(ids, skip_special_tokens=true)`
- `decode_batch(sequences, skip_special_tokens=true)`
- streaming decode with `DecodeStream::step(id)` and `has_pending()`
- `token_to_id`, `id_to_token`, and `get_vocab_size`
- narrow runtime mutation APIs used by accepted documentation and stream
  parity slices: `add_tokens`, `with_byte_level_normalizer`,
  `with_split_pre_tokenizer`, and `with_wordpiece_decoder`

`Encoding` exposes ids, type ids, tokens, offsets, word ids, special-token
masks, attention masks, and overflowing encodings.

## Runtime Coverage

The completed core covers the inference paths needed by the accepted real
tokenizer and upstream-derived parity suite:

- tokenizer JSON load and wrapper dispatch
- added tokens and special tokens
- normalizers, including ICU-backed Unicode normalization and regex replace
- pre-tokenizers, including ByteLevel, Bert, Whitespace, Split, Digits, and
  Metaspace/SentencePiece-style paths where covered by real fixtures
- models: WordLevel, WordPiece, BPE, and Unigram inference
- post-processors: BertProcessing, RobertaProcessing, TemplateProcessing, and
  PostProcessor Sequence for covered shapes
- decoders: ByteLevel, WordPiece, Metaspace, ByteFallback, CTC, Fuse, and
  decoder Sequence for covered shapes
- truncation, padding, and overflowing encodings
- ordered batch encode/decode execution
- stream decode for the accepted ByteLevel, Metaspace, and ByteFallback slices
- vendored static ICU4C as the default Unicode backend, with a Linux audit that
  rejects accidental shared `libicu*.so` runtime linkage

## Real Tokenizer Smoke Matrix

The downstream-facing smoke suite loads local HF test-data tokenizer JSON files
and verifies ids, tokens, offsets, type ids, word ids, masks, overflowing where
applicable, and decode behavior.

| Family | Fixture | Covered public entry shape |
| --- | --- | --- |
| GPT-style ByteLevel BPE | `tokenizer.json` temporary Sequence shape | single, batch, pair-batch, decode-batch, pre-tokenized batch slices |
| RoBERTa ByteLevel BPE | `roberta.json` | single and ordered pair-batch |
| BERT WordPiece | `bert-wiki.json` | single, ordered pair-batch, WordPiece decode-batch |
| ALBERT/SentencePiece Unigram | `albert-base-v1-tokenizer.json` | single, truncation, padding, overflowing, batch, decode-batch |
| Llama Split + ByteLevel BPE | `llama-3-tokenizer.json` | single, ordered batch, ordered pair-batch, decode-batch |

## Verification Baseline

The milestone baseline is the default vendored-ICU build:

```sh
cmake --build projects/tokenizers.cpp/build-icu
ctest --test-dir projects/tokenizers.cpp/build-icu --output-on-failure
uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py --check
```

Latest freeze result:

- C++ build passed.
- CTest passed: 26/26.
- Fixture generator check passed.
- `third_party/tokenizers` and `third_party/icu` worktrees were clean.

## Exclusions

These are intentionally not included in the completed core inference milestone:

- trainers and training tests
- HTTP/from-pretrained remote loading
- benches as acceptance criteria
- Python, Node, and WASM wrappers
- Rust FFI or runtime shell-out
- Unigram sampling
- full upstream internal model-builder API parity
- full public `models::BPE`/builder/cache API parity
- exact stochastic BPE dropout parity beyond the accepted deterministic and
  shape-invariant boundaries

## Hardening Queue

The next phase is hardening, not core translation:

- BPE lower-level public surface is decided by `ADR-0004`: stay
  tokenizer-centered through `Tokenizer::from_file` and
  `Tokenizer::from_bpe_files`; re-open only if downstream consumers need
  lower-level APIs.
- Additional SentencePiece `Precompiled` charsmap validation when new real
  tokenizer JSON fixtures demand it. R4-H1 broadened the local ALBERT coverage
  for no-break/BOM boundaries, chained zero-width deletion, compatibility
  expansion, and expansion followed by joiner splitting.
- Additional mixed decoder and stream edge cases when demanded by generation
  consumers. The first R4 hardening pass pins ByteFallback/Fuse,
  Metaspace/Fuse, ByteLevel/Fuse, and the no-finalize reset policy.
- Rare ICU offset projection cases, especially unusual canonical reordering.
- Additional real tokenizer smoke fixtures as downstream model projects require
  them.

The default public API remains tokenizer-centered. Lower-level component APIs
should stay private unless a concrete downstream integration needs them.
