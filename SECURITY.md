# Security Policy

`tokenizers.cpp` is an inference-only tokenizer runtime. It does not perform
network loading, training, or execution of tokenizer-provided code.

Please report security issues privately to the project maintainers before
opening a public issue.

When reporting a vulnerability, include:

- affected commit or release
- platform and compiler
- tokenizer JSON or minimal reproducer
- expected and observed behavior
- whether the issue requires untrusted tokenizer JSON, untrusted input text, or
  both
