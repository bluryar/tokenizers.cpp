# Real Tokenizer Smoke Suite

## Scope

This suite is a downstream-facing confidence layer over the existing focused
parity tests. It does not replace component-specific tests; it verifies that the
main real tokenizer JSON families still compose through the public C++ API.

The suite remains native C++ inference only. It must not add Rust FFI, runtime
shell-out, training, HTTP/from-pretrained loading, or edits under
`third_party/tokenizers`.

## Covered Matrix

`tests/parity/real_tokenizer_smoke_test.cpp` loads local HF test-data fixtures
and verifies ids, tokens, offsets, type ids, word ids, special-token masks,
attention masks, and decode behavior for:

- GPT-style ByteLevel BPE from `tokenizer.json`, with the accepted
  `Sequence(ByteLevel, TemplateProcessing)` post-processor shape used by the
  post-processor sequence parity slice, including ordered `encode_batch`,
  ordered `encode_batch_pairs`, and `decode_batch` over the same real-JSON
  path.
- RoBERTa ByteLevel BPE from real `roberta.json`, including ordered
  `encode_batch_pairs` over the real post-processor shape.
- BERT WordPiece from real `bert-wiki.json`, including ordered
  `encode_batch_pairs` over the real TemplateProcessing pair shape and the
  explicit `with_wordpiece_decoder()` cleanup transition.
- ALBERT/SentencePiece-style Unigram from real
  `albert-base-v1-tokenizer.json`, including a temporary real-JSON copy with
  truncation and fixed padding that verifies main and overflowing encodings plus
  skip-special and full decode behavior, plus ordered batch encode and
  `decode_batch` over main and overflowing ids.
- Llama Split + ByteLevel BPE from real `llama-3-tokenizer.json`, including
  ordered `encode_batch`, ordered `encode_batch_pairs`, and `decode_batch`
  over regular and pair outputs.

## Boundaries

- Expected values are intentionally small and stable. Deep edge cases stay in
  the focused component tests.
- The GPT-style entry mutates a temporary copy of `tokenizer.json` to install
  the already accepted `Sequence(ByteLevel, TemplateProcessing)` shape. This
  keeps upstream fixtures read-only while exercising the production
  `Tokenizer::from_file` path.
- This suite is a smoke layer. If a field regresses here, debug in the relevant
  component parity spec first.
