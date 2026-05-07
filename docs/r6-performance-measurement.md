# R6 Performance Measurement

Date: 2026-05-07

## Status

R6 benchmark/measurement coverage now includes both self-contained synthetic
hot-path cases and HF-test-data-backed real tokenizer JSON cases.

## Decision

Performance work stays behind the tokenizer-centered public API. R6 now has two
optional benchmark executables:

- `tokenizers_cpp_added_token_matcher_benchmark`: direct legacy-scan versus
  Aho-Corasick candidate collection for added-token matching.
- `tokenizers_cpp_r6_performance_benchmark`: public-API runtime matrix for the
  current R6 hot paths.

Benchmarks are not CTest assertions and are not built by default.

## Runtime Matrix

`tokenizers_cpp_r6_performance_benchmark` covers:

- `added_tokens_1024`: tokenizer-level added-token-heavy encode path.
- `bpe_cache_repeated_hello`: repeated short BPE pieces that should benefit
  from the private BPE thread-local cache.
- `bpe_heap_long_piece_512`: a deterministic long raw BPE piece larger than the
  cache cutoff, exercising heap merge selection directly.
- `unigram_trie_cache_ab`: repeated Unigram pieces through `WhitespaceSplit`,
  exercising trie lookup plus thread-local cache.

These cases use self-contained temporary tokenizer JSON files and do not require
HF test data.

When `TOKENIZERS_CPP_HF_TEST_DATA_DIR` exists at configure time, the benchmark
also adds real tokenizer JSON cases:

- `real_gpt_sequence_batch`: GPT-style ByteLevel BPE plus
  `Sequence(ByteLevel, TemplateProcessing)` batch encode.
- `real_roberta_pair_batch`: RoBERTa ByteLevel BPE pair-batch encode.
- `real_bert_batch_decode`: BERT WordPiece batch encode plus decode.
- `real_albert_batch_decode`: ALBERT/SentencePiece Unigram batch encode plus
  decode.
- `real_llama_pair_batch`: Llama-style Split/ByteLevel BPE pair-batch encode.

## Commands

```sh
cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build-bench \
  -DTOKENIZERS_CPP_BUILD_TESTS=OFF \
  -DTOKENIZERS_CPP_BUILD_EXAMPLES=OFF \
  -DTOKENIZERS_CPP_BUILD_BENCHMARKS=ON \
  -DTOKENIZERS_CPP_FETCH_DEPS=OFF
cmake --build projects/tokenizers.cpp/build-bench \
  --target tokenizers_cpp_r6_performance_benchmark \
           tokenizers_cpp_added_token_matcher_benchmark
projects/tokenizers.cpp/build-bench/tokenizers_cpp_r6_performance_benchmark
projects/tokenizers.cpp/build-bench/tokenizers_cpp_added_token_matcher_benchmark
```

## Current Local Output

`tokenizers_cpp_r6_performance_benchmark`:

```text
added_tokens_1024                  iterations=100   input_bytes=10496    elapsed_ms=958.140    MiB/s=1.045      enc/s=104.369    checksum=14154630344326105512
bpe_cache_repeated_hello           iterations=200   input_bytes=12287    elapsed_ms=721.499    MiB/s=3.248      enc/s=277.200    checksum=4800127977224962048
bpe_heap_long_piece_512            iterations=500   input_bytes=512      elapsed_ms=278.611    MiB/s=0.876      enc/s=1794.614   checksum=3779030594713542080
unigram_trie_cache_ab              iterations=200   input_bytes=12287    elapsed_ms=817.392    MiB/s=2.867      enc/s=244.681    checksum=4519017675076698112
real_gpt_sequence_batch            iterations=120   input_bytes=352      elapsed_ms=96.737     MiB/s=0.416      enc/s=1240.482   checksum=6339901532157670672
real_roberta_pair_batch            iterations=80    input_bytes=408      elapsed_ms=60.058     MiB/s=0.518      enc/s=1332.056   checksum=1852651245518534752
real_bert_batch_decode             iterations=80    input_bytes=950      elapsed_ms=65.360     MiB/s=1.109      enc/s=1223.988   checksum=4713035283500748992
real_albert_batch_decode           iterations=80    input_bytes=696      elapsed_ms=85.321     MiB/s=0.622      enc/s=937.641    checksum=4970984985901250128
real_llama_pair_batch              iterations=60    input_bytes=780      elapsed_ms=51.608     MiB/s=0.865      enc/s=1162.619   checksum=6112667824444849188
```

`tokenizers_cpp_added_token_matcher_benchmark`:

```text
tokens=16 repeats=64 legacy_ms=7.76516 trie_ms=5.62151 speedup=1.38133x
tokens=128 repeats=128 legacy_ms=559.774 trie_ms=11.4814 speedup=48.7549x
tokens=1024 repeats=256 legacy_ms=22745.9 trie_ms=22.6864 speedup=1002.63x
```

## Next Measurement Gate

Use this matrix as the baseline for future profile-driven R6 decisions.
