# AGENTS.md

## Mission

`tokenizers.cpp` is a native C++ inference-only porting workspace for the
Hugging Face `tokenizers` Rust crate.

The goal is to translate the runtime tokenizer behavior needed for inference
into C++ while keeping parity with upstream tests from
`third_party/tokenizers/tokenizers`. This is not a Rust FFI wrapper and not a
training project.

## Sources Of Truth

- Upstream reference repo:
  `projects/tokenizers.cpp/third_party/tokenizers`
- Upstream Rust crate focus:
  `projects/tokenizers.cpp/third_party/tokenizers/tokenizers`
- Project API and implementation:
  `projects/tokenizers.cpp/include/tokenizers_cpp`
  and `projects/tokenizers.cpp/src`
- Parity tests:
  `projects/tokenizers.cpp/tests/parity`

When these disagree, prefer:

1. Upstream `tokenizers/tokenizers` runtime behavior and tests for tokenizer
   semantics.
2. This project's ADRs/specs for accepted native C++ boundaries.
3. Root `ggbond` policy for project hygiene and agent state.

## Working Rules

- Implement native C++ runtime behavior only. Do not bind to Rust, call Rust
  binaries from the runtime, or add a Rust FFI dependency.
- Exclude trainers, training tests, HTTP/from-pretrained network loading,
  benches, and Python/Node/WASM wrapper work unless a later task explicitly
  changes scope.
- Treat `third_party/tokenizers` as a read-only reference. Do not patch it.
- Preserve Hugging Face runtime field names and semantics where the public C++
  API mirrors them: ids, type ids, tokens, offsets, word ids, special-token
  masks, attention masks, overflowing encodings, truncation, and padding.
- Keep test parity decisions on disk in `docs/specs/`, not only in chat.
- Use CMake for this project. Keep dependency versions pinned in CMake/docs.

## Verification

- Default project smoke:
  `cmake -S projects/tokenizers.cpp -B projects/tokenizers.cpp/build -DTOKENIZERS_CPP_BUILD_TESTS=ON`
  `cmake --build projects/tokenizers.cpp/build`
  `ctest --test-dir projects/tokenizers.cpp/build --output-on-failure`
- Upstream Rust tests are reference checks only. If Rust is missing or a test
  requires training/http features, record the blocker or exclusion explicitly.

## Shared State Files

- `docs/vision.md`: project goal, non-goals, and scope boundaries
- `docs/roadmap.md`: milestone order and dependency graph
- `docs/status.md`: current upstream ref, blockers, and next actions
- `docs/adr/*.md`: architecture and policy decisions
- `docs/specs/*.md`: parity specs, test inventories, and execution plans

## Local Agents

- Prefer project-local agents in `.codex/agents/` when working from this
  directory.
- Keep root `ggbond` agents for shared platform work only.
