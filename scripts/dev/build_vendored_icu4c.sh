#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
icu_source_dir="${TOKENIZERS_CPP_ICU_SOURCE_DIR:-${repo_root}/third_party/icu/icu4c/source}"
icu_build_dir="${TOKENIZERS_CPP_ICU_BUILD_DIR:-${repo_root}/build/icu4c-static}"
icu_install_dir="${TOKENIZERS_CPP_ICU_INSTALL_DIR:-${repo_root}/third_party/icu4c-install}"
jobs="${TOKENIZERS_CPP_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

if [[ ! -x "${icu_source_dir}/runConfigureICU" ]]; then
  cat >&2 <<EOF
ICU4C source was not found at:
  ${icu_source_dir}

Expected a vendored unicode-org/icu checkout or extracted release containing
icu4c/source/runConfigureICU. Set TOKENIZERS_CPP_ICU_SOURCE_DIR to override.
EOF
  exit 1
fi

uv_python_exe="$(
  uv run --no-project python -c 'import sys; print(sys.executable)' 2>/dev/null || true
)"
if [[ -n "${uv_python_exe}" && -x "${uv_python_exe}" ]]; then
  export PATH="$(dirname "${uv_python_exe}"):${PATH}"
fi

mkdir -p "${icu_build_dir}" "${icu_install_dir}"
cd "${icu_build_dir}"

"${icu_source_dir}/runConfigureICU" Linux \
  --prefix="${icu_install_dir}" \
  --enable-static \
  --disable-shared \
  --with-data-packaging=archive

make -j"${jobs}"
make install

cat <<EOF
Vendored ICU4C installed to:
  ${icu_install_dir}

Configure tokenizers.cpp with:
  cmake -S ${repo_root} -B ${repo_root}/build-icu

If the install prefix is not the default, pass:
  -DTOKENIZERS_CPP_ICU_ROOT=${icu_install_dir}
EOF
