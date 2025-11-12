#!/usr/bin/env bash
set -euo pipefail

CRATE="deepseek-ocr-android"
TARGET="${TARGET:-aarch64-linux-android}"
ABI="${ABI:-arm64-v8a}"
PROFILE="${PROFILE:-release}"
OUT_DIR="${OUT_DIR:-target/android}"
UNIFFI_BINDGEN_BIN="${UNIFFI_BINDGEN_BIN:-uniffi-bindgen}"
UNIFFI_OUTPUT_ROOT="${UNIFFI_OUTPUT_ROOT:-bindings}"

if [[ "${PROFILE}" == "release" ]]; then
  CARGO_FLAGS=(--release)
else
  CARGO_FLAGS=(--profile "${PROFILE}")
fi

artifact_path() {
  local base="$1"
  printf "%s/libdeepseek_ocr_android.so" "$base"
}

find_existing_artifact() {
  local candidates=("$@")
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  find target -name "libdeepseek_ocr_android.so" -print -quit
}

if command -v cargo-ndk >/dev/null 2>&1; then
  echo "[build-android] Using cargo-ndk (ABI=${ABI}, profile=${PROFILE})"
  cargo ndk -t "${ABI}" -o "${OUT_DIR}" build -p "${CRATE}" "${CARGO_FLAGS[@]}"
  ARTIFACT="$(find_existing_artifact \
    "$(artifact_path "${OUT_DIR}/${ABI}/${PROFILE}")" \
    "$(artifact_path "${OUT_DIR}/${ABI}")")"
else
  echo "[build-android] cargo-ndk not found; falling back to cargo build for target ${TARGET}"
  if ! rustup target list --installed | grep -q "^${TARGET}$"; then
    rustup target add "${TARGET}"
  fi
  cargo build -p "${CRATE}" --target "${TARGET}" "${CARGO_FLAGS[@]}"
  ARTIFACT="$(find_existing_artifact "$(artifact_path "target/${TARGET}/${PROFILE}")")"
fi

if [[ -f "${ARTIFACT}" ]]; then
  echo "[build-android] ✅ Shared library ready at ${ARTIFACT}"
else
  echo "[build-android] ⚠️ Build finished but ${ARTIFACT} was not found" >&2
  exit 1
fi

if command -v "${UNIFFI_BINDGEN_BIN}" >/dev/null 2>&1; then
  OUT_PATH="${UNIFFI_OUTPUT_ROOT}/kotlin"
  mkdir -p "${OUT_PATH}"
  echo "[build-android] Generating UniFFI Kotlin bindings into ${OUT_PATH}"
  "${UNIFFI_BINDGEN_BIN}" generate \
    --library "${ARTIFACT}" \
    --language kotlin \
    --out-dir "${OUT_PATH}"
else
  echo "[build-android] ⚠️ UniFFI generation skipped: ${UNIFFI_BINDGEN_BIN} not found" >&2
fi
