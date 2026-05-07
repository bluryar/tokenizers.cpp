# /// script
# requires-python = ">=3.10"
# ///
"""Print a lightweight inventory of upstream Rust tests.

Run from the repository root with:

    uv run --no-project --script projects/tokenizers.cpp/scripts/dev/inventory_upstream_tests.py
"""

from __future__ import annotations

import pathlib
import re

PROJECT = pathlib.Path(__file__).resolve().parents[2]
TESTS = PROJECT / "third_party" / "tokenizers" / "tokenizers" / "tests"


def classify(path: pathlib.Path, name: str) -> str:
    rel = path.name
    lower = f"{rel}::{name}".lower()
    if "training" in lower or "train_" in lower or name in {
        "train_tokenizer",
        "quicktour_slow_train",
        "train_pipeline_bert",
        "test_train_unigram_from_file",
    }:
        return "skip-training"
    if rel == "from_pretrained.rs":
        return "skip-non-core"
    if rel == "common":
        return "reference-only"
    if name == "test_sample":
        return "reference-only"
    return "port"


def main() -> None:
    if not TESTS.exists():
        raise SystemExit(f"missing upstream tests directory: {TESTS}")

    print("| Upstream test | Classification |")
    print("| --- | --- |")
    for path in sorted(TESTS.glob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?m)^fn\s+([A-Za-z0-9_]+)\s*\(", text):
            name = match.group(1)
            print(f"| `{path.name}::{name}` | {classify(path, name)} |")


if __name__ == "__main__":
    main()
