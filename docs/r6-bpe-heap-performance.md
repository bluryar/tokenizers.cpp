# R6 BPE Merge Heap Performance

Date: 2026-05-07

## Status

R6 deterministic BPE merge heap performance hardening is complete.

## Decision

The deterministic BPE merge loop now uses a private priority queue of adjacent
merge candidates. Initial candidates are built once, and after a merge only the
two affected neighboring pairs are rechecked.

This replaces the previous deterministic loop that rescanned every adjacent
pair after every merge.

## Boundary

- The public tokenizer-centered API does not change.
- Stochastic dropout paths still use the previous linear scan so random
  candidate-skip behavior does not drift.
- `dropout=null` and `dropout=0.0` use the heap path.
- Cache behavior remains separate: repeated pieces can still bypass merge work
  through the R6 BPE thread-local cache.
- Priority-queue candidates carry node generations, so stale candidates from
  earlier adjacency states are ignored.

## Rust Reference

Rust upstream uses heap-oriented merge work to avoid repeatedly scanning all
pairs in long BPE pieces. This C++ slice applies the same performance idea while
keeping model internals private.

## Validation

- `tokenizers_cpp_bpe_config_test` covers deterministic merge parity, dropout
  boundaries, and a stale-candidate heap fixture where an old lower-rank pair
  must be ignored after a neighboring merge changes adjacency.
- Full ICU CTest and no-HF open-source smoke pass with the heap path enabled.
