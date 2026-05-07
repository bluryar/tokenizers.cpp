# /// script
# requires-python = ">=3.10"
# ///
"""Generate development-time parity fixtures from upstream Rust tokenizers.

This script is intentionally outside the C++ runtime. It builds a temporary
Cargo helper against the read-only upstream checkout and records reference
encoding data as JSON fixtures for C++ parity tests.

Run from the repository root with:

    uv run --no-project --script projects/tokenizers.cpp/scripts/dev/generate_parity_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile
import textwrap

PROJECT = pathlib.Path(__file__).resolve().parents[2]
UPSTREAM_TOKENIZERS = PROJECT / "third_party" / "tokenizers" / "tokenizers"
FIXTURES = PROJECT / "tests" / "parity" / "fixtures"

TOKENIZER_JSON = FIXTURES / "simple_wordlevel_tokenizer.json"
FIXTURE_JSON = FIXTURES / "simple_wordlevel_fixture.json"


SIMPLE_WORDLEVEL_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 1,
            "content": "[CLS]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "WhitespaceSplit"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "[UNK]": 0,
            "[CLS]": 1,
            "hello": 2,
            "world": 3,
            "goodbye": 4,
        },
        "unk_token": "[UNK]",
    },
}


RUST_MAIN = r'''
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Serialize)]
struct Offset {
    start: usize,
    end: usize,
}

#[derive(Serialize)]
struct EncodingFixture {
    ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<Offset>,
    type_ids: Vec<u32>,
    word_ids: Vec<Option<u32>>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
}

#[derive(Serialize)]
struct CaseFixture {
    name: String,
    text: String,
    add_special_tokens: bool,
    encoding: EncodingFixture,
    decode: String,
}

#[derive(Serialize)]
struct Fixture {
    source: String,
    tokenizer_json: String,
    cases: Vec<CaseFixture>,
}

fn encoding_fixture(encoding: &tokenizers::Encoding) -> EncodingFixture {
    EncodingFixture {
        ids: encoding.get_ids().to_vec(),
        tokens: encoding.get_tokens().to_vec(),
        offsets: encoding
            .get_offsets()
            .iter()
            .map(|(start, end)| Offset { start: *start, end: *end })
            .collect(),
        type_ids: encoding.get_type_ids().to_vec(),
        word_ids: encoding.get_word_ids().to_vec(),
        special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
        attention_mask: encoding.get_attention_mask().to_vec(),
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let tokenizer_path = PathBuf::from(args.next().expect("tokenizer path"));
    let fixture_path = PathBuf::from(args.next().expect("fixture path"));

    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
    let mut cases = Vec::new();
    for (name, text, add_special_tokens) in [
        ("hello_world", "hello world", true),
        ("leading_and_repeated_space", "  hello   world", true),
    ] {
        let encoding = tokenizer.encode(text, add_special_tokens).expect("encode");
        let decode = tokenizer
            .decode(encoding.get_ids(), true)
            .expect("decode encoded ids");
        cases.push(CaseFixture {
            name: name.to_string(),
            text: text.to_string(),
            add_special_tokens,
            encoding: encoding_fixture(&encoding),
            decode,
        });
    }

    let fixture = Fixture {
        source: "upstream tokenizers Rust crate via scripts/dev/generate_parity_fixtures.py".to_string(),
        tokenizer_json: tokenizer_path
            .file_name()
            .expect("tokenizer filename")
            .to_string_lossy()
            .to_string(),
        cases,
    };

    let bytes = serde_json::to_vec_pretty(&fixture).expect("serialize fixture");
    fs::write(fixture_path, bytes).expect("write fixture");
}
'''


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run_rust_reference() -> None:
    if not UPSTREAM_TOKENIZERS.exists():
        raise SystemExit(f"missing upstream tokenizers checkout: {UPSTREAM_TOKENIZERS}")

    with tempfile.TemporaryDirectory(prefix="tokenizers_cpp_fixture_") as temp_dir:
        temp = pathlib.Path(temp_dir)
        (temp / "src").mkdir()
        (temp / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "tokenizers_cpp_fixture_generator"
                version = "0.1.0"
                edition = "2021"

                [dependencies]
                tokenizers = {{ path = {json.dumps(str(UPSTREAM_TOKENIZERS))}, default-features = false, features = ["fancy-regex"] }}
                serde = {{ version = "1", features = ["derive"] }}
                serde_json = "1"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (temp / "src" / "main.rs").write_text(RUST_MAIN, encoding="utf-8")

        subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(temp / "Cargo.toml"),
                "--",
                str(TOKENIZER_JSON),
                str(FIXTURE_JSON),
            ],
            check=True,
            cwd=PROJECT,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated fixtures differ from the committed files",
    )
    args = parser.parse_args()

    if shutil.which("cargo") is None:
        raise SystemExit("cargo is required to generate Rust reference fixtures")

    before = {
        path: path.read_bytes() if path.exists() else None
        for path in (TOKENIZER_JSON, FIXTURE_JSON)
    }
    write_json(TOKENIZER_JSON, SIMPLE_WORDLEVEL_TOKENIZER)
    run_rust_reference()

    if args.check:
        changed = [
            str(path.relative_to(PROJECT))
            for path, previous in before.items()
            if previous != path.read_bytes()
        ]
        if changed:
            raise SystemExit("fixtures are not up to date: " + ", ".join(changed))


if __name__ == "__main__":
    main()
