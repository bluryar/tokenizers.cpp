# tokenizers.cpp Vision

## Goal

Build a native C++ inference-only implementation of Hugging Face tokenizer
runtime behavior inside `ggbond`.

The project uses the Rust `tokenizers/tokenizers` crate as the parity reference
and translates its runtime behavior and tests into a C++ library that future
GGML/C++ inference projects can depend on without Python, Rust FFI, or network
runtime loading.

## Non-Goals

- No tokenizer training or trainer APIs
- No Rust FFI or runtime shell-out to Rust tools
- No HTTP/from-pretrained model download surface
- No benches as acceptance criteria
- No Python, Node, or WASM wrapper implementation

## Scope Boundaries

| Area | In Scope | Out Of Scope |
| --- | --- | --- |
| Loading | Local tokenizer JSON files | Remote model fetching |
| Runtime | encode, encode pair, decode, streaming decode where needed | training |
| Components | normalizers, pre-tokenizers, models, processors, decoders, added tokens | trainer implementations |
| Models | BPE, WordPiece, WordLevel, Unigram inference | model training algorithms |
| Validation | C++ parity tests translated from upstream Rust tests | upstream benchmark parity |
