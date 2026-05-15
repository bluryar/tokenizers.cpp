#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
icu_version="${TOKENIZERS_CPP_ICU_VERSION:-78.3}"
icu_archive_dir="${TOKENIZERS_CPP_ICU_ARCHIVE_DIR:-${repo_root}/third_party/icu4c-${icu_version}}"
icu_source_archive="${TOKENIZERS_CPP_ICU_SOURCE_ARCHIVE:-${icu_archive_dir}/icu4c-${icu_version}-sources.tgz}"
icu_data_archive="${TOKENIZERS_CPP_ICU_DATA_ARCHIVE:-${icu_archive_dir}/icu4c-${icu_version}-data.zip}"
icu_extract_dir="${TOKENIZERS_CPP_ICU_EXTRACT_DIR:-${repo_root}/build/icu4c-${icu_version}-source}"
icu_source_dir="${TOKENIZERS_CPP_ICU_SOURCE_DIR:-${icu_extract_dir}/icu/source}"
icu_build_dir="${TOKENIZERS_CPP_ICU_BUILD_DIR:-${repo_root}/build/icu4c-static}"
icu_install_dir="${TOKENIZERS_CPP_ICU_INSTALL_DIR:-${repo_root}/third_party/icu4c-install}"
jobs="${TOKENIZERS_CPP_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

if [[ ! -x "${icu_source_dir}/runConfigureICU" ]]; then
  if [[ -f "${icu_source_archive}" ]]; then
    rm -rf "${icu_extract_dir}"
    mkdir -p "${icu_extract_dir}"
    tar -xzf "${icu_source_archive}" -C "${icu_extract_dir}"
  fi
fi

if [[ ! -x "${icu_source_dir}/runConfigureICU" ]]; then
  cat >&2 <<EOF
ICU4C source was not found at:
  ${icu_source_dir}

Expected the vendored ICU release archive at:
  ${icu_source_archive}

or an extracted release containing icu/source/runConfigureICU. Set
TOKENIZERS_CPP_ICU_SOURCE_DIR to override.
EOF
  exit 1
fi

if [[ ! -f "${icu_data_archive}" ]]; then
  cat >&2 <<EOF
ICU4C data archive was not found at:
  ${icu_data_archive}

The source archive contains buildable ICU data, but tokenizers.cpp vendors the
matching release data archive too so release provenance is explicit.
EOF
  exit 1
fi

uv_python_exe="$(
  uv run --no-project python -c 'import sys; print(sys.executable)' 2>/dev/null || true
)"
if [[ -n "${uv_python_exe}" && -x "${uv_python_exe}" ]]; then
  export PATH="$(dirname "${uv_python_exe}"):${PATH}"
fi

rm -rf "${icu_build_dir}" "${icu_install_dir}"
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

Vendored ICU release inputs:
  source: ${icu_source_archive}
  data:   ${icu_data_archive}

Configure tokenizers.cpp with:
  cmake -S ${repo_root} -B ${repo_root}/build-icu

If the install prefix is not the default, pass:
  -DTOKENIZERS_CPP_ICU_ROOT=${icu_install_dir}
EOF
