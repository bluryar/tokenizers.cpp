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
- `wordpiece_trie_repeated`: repeated WordPiece pieces through `Whitespace`,
  exercising private initial/continuation trie lookup.
- `wordpiece_small_batch`: small WordPiece batch encode that should stay serial
  after the batch parallelism threshold.

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
added_tokens_1024                  iterations=100   input_bytes=10496    elapsed_ms=325.926    MiB/s=3.071      enc/s=306.818    checksum=14154630344326105512
bpe_cache_repeated_hello           iterations=200   input_bytes=12287    elapsed_ms=679.972    MiB/s=3.447      enc/s=294.130    checksum=4800127977224962048
bpe_heap_long_piece_512            iterations=500   input_bytes=512      elapsed_ms=272.153    MiB/s=0.897      enc/s=1837.203   checksum=3779030594713542080
unigram_trie_cache_ab              iterations=200   input_bytes=12287    elapsed_ms=803.209    MiB/s=2.918      enc/s=249.001    checksum=4519017675076698112
wordpiece_trie_repeated            iterations=200   input_bytes=16895    elapsed_ms=731.127    MiB/s=4.408      enc/s=273.550    checksum=11292211218858008576
wordpiece_small_batch              iterations=1000  input_bytes=68       elapsed_ms=47.324     MiB/s=1.370      enc/s=21130.720  checksum=11722130037619660296
real_gpt_sequence_batch            iterations=120   input_bytes=352      elapsed_ms=69.223     MiB/s=0.582      enc/s=1733.533   checksum=6339901532157670672
real_roberta_pair_batch            iterations=80    input_bytes=408      elapsed_ms=42.879     MiB/s=0.726      enc/s=1865.718   checksum=1852651245518534752
real_bert_batch_decode             iterations=80    input_bytes=950      elapsed_ms=56.583     MiB/s=1.281      enc/s=1413.844   checksum=4713035283500748992
real_albert_batch_decode           iterations=80    input_bytes=696      elapsed_ms=57.437     MiB/s=0.924      enc/s=1392.828   checksum=4970984985901250128
real_llama_pair_batch              iterations=60    input_bytes=780      elapsed_ms=49.040     MiB/s=0.910      enc/s=1223.499   checksum=6112667824444849188
```

`tokenizers_cpp_added_token_matcher_benchmark`:

```text
tokens=16 repeats=64 legacy_ms=7.76516 trie_ms=5.62151 speedup=1.38133x
tokens=128 repeats=128 legacy_ms=559.774 trie_ms=11.4814 speedup=48.7549x
tokens=1024 repeats=256 legacy_ms=22745.9 trie_ms=22.6864 speedup=1002.63x
```

## Next Measurement Gate

Use this matrix as the baseline for future profile-driven R6 decisions.
