# tokenizers.cpp Inference Parity Spec

## Goal

Translate Hugging Face `tokenizers/tokenizers` inference behavior into native
C++ and validate it with C++ parity tests derived from upstream Rust tests.

## Public C++ API

- `Tokenizer::from_file(path)`
- `Tokenizer::from_bpe_files(vocab_path, merges_path, options={})`
- `Tokenizer::encode(text, add_special_tokens=true)`
- `Tokenizer::encode(pre_tokenized_words, add_special_tokens=true)` for the
  conservative `std::vector<std::string>` pre-tokenized boundary
- `Tokenizer::encode_char_offsets(text, add_special_tokens=true)`
- `Tokenizer::encode_char_offsets(pre_tokenized_words, add_special_tokens=true)`
- `Tokenizer::encode_pair(text_a, text_b, add_special_tokens=true)`
- `Tokenizer::encode_pair(pre_tokenized_a, pre_tokenized_b, add_special_tokens=true)`
- `Tokenizer::encode_pair_char_offsets(text_a, text_b, add_special_tokens=true)`
- `Tokenizer::encode_pair_char_offsets(pre_tokenized_a, pre_tokenized_b, add_special_tokens=true)`
- `Tokenizer::encode_batch(texts, add_special_tokens=true)`
- `Tokenizer::encode_batch(pre_tokenized_texts, add_special_tokens=true)`
- `Tokenizer::encode_batch_char_offsets(texts, add_special_tokens=true)`
- `Tokenizer::encode_batch_char_offsets(pre_tokenized_texts, add_special_tokens=true)`
- `Tokenizer::encode_batch_pairs(pairs, add_special_tokens=true)`
- `Tokenizer::encode_batch_pairs(pre_tokenized_pairs, add_special_tokens=true)`
- `Tokenizer::encode_batch_pairs_char_offsets(pairs, add_special_tokens=true)`
- `Tokenizer::encode_batch_pairs_char_offsets(pre_tokenized_pairs, add_special_tokens=true)`
- `Tokenizer::decode(ids, skip_special_tokens=true)`
- `Tokenizer::decode_batch(sequences, skip_special_tokens=true)`
- `Tokenizer::decode_stream(skip_special_tokens=true)` with
  `DecodeStream::step(id)` and `DecodeStream::has_pending()`
- `Tokenizer::token_to_id(token)`
- `Tokenizer::id_to_token(id)`
- `Tokenizer::get_vocab_size()`

`Encoding` must expose:

- `ids`
- `type_ids`
- `tokens`
- `offsets`
- `word_ids`
- `special_tokens_mask`
- `attention_mask`
- `overflowing`

## Runtime Components

Port these inference paths:

- tokenizer JSON load and wrapper type dispatch
- added tokens and special tokens
- normalizers
- pre-tokenizers
- BPE, WordPiece, WordLevel, and Unigram inference models
- post-processors
- decoders, including streaming decode if needed by downstream generation
- truncation and padding
- ordered batch encode/decode execution
- explicit char-offset encode APIs

## Exclusions

- trainer APIs and training tests
- HTTP/from-pretrained remote loading
- benches
- Python, Node, and WASM wrappers
- Rust FFI or runtime shell-out
- standalone public model/component builder APIs, including public
  `models::BPE`, unless a downstream-driven ADR revises the public surface

## Acceptance Criteria

- Every upstream Rust test in `tokenizers/tests` is classified in the test
  inventory.
- Tests classified as `port` have corresponding C++ parity coverage or a
  recorded blocker.
- C++ tests verify ids, tokens, offsets, type ids, word ids, masks, overflowing
  encodings, and decoded text when the upstream test exposes them.
- Real tokenizer JSON smoke fixtures cover GPT/BERT/Roberta/SentencePiece-style
  tokenizers before calling the runtime consumer-ready.
