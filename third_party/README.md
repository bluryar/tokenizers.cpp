# Third-Party Submodules And Local Artifacts

`third_party/tokenizers/` is a git submodule for
`https://github.com/huggingface/tokenizers.git`.

It is treated as a read-only upstream reference for parity, fixtures, and
source-level comparison. Do not patch it as part of the native C++ port. Record
the exact checked-out commit in `docs/status.md` whenever the submodule pin is
created or updated.

`third_party/icu/` is a git submodule for
`https://github.com/unicode-org/icu.git`.

ICU4C uses a project-owned static install generated from that source:

- `third_party/icu/icu4c/source` is built by
  `scripts/dev/build_vendored_icu4c.sh`
- `third_party/icu4c-install` for the resulting static install prefix consumed
  by `TOKENIZERS_CPP_ICU_ROOT`

`third_party/icu4c-install` is a local build artifact and remains ignored by
git. It must not be committed.

Current Hugging Face tokenizers source pin:

- Repository: `https://github.com/huggingface/tokenizers.git`
- Commit: `22d54d37621f2d9f35cf9420d6ed8658372a6c5d`

Current local ICU4C source pin:

- Repository: `https://github.com/unicode-org/icu.git`
- Tag: `release-77-1`
- Commit: `457157a92aa053e632cc7fcfd0e12f8a943b2d11`
- ICU4C version: `77.1`
- Unicode version: `16.0`

Do not let the default CMake build discover system ICU.
