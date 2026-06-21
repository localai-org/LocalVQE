#!/usr/bin/env bash
#
# apply_ggml_patches.sh
#
# Apply the in-tree ggml patches to vendor/ggml. Idempotent: re-running is a
# no-op once everything is applied.
#
# We keep the ggml submodule pinned to its upstream commit and carry our
# local changes (currently the fused GGML_OP_GRU op the GTCRN backend needs)
# as numbered patch files in vendor/ggml-patches/, applied in filename order
# (the numeric prefix from `git format-patch` gives the right ordering).
#
# Run this once after `git submodule update --init`, before configuring the
# C++ build. The Docker cross-build (docker/Dockerfile.arm64) copies an
# already-patched tree, so it does not need to call this.
#
# Usage:
#   bash ggml/scripts/apply_ggml_patches.sh
#
# Exits 0 on success, non-zero on any failure.

set -euo pipefail

# Resolve paths from the script's own location so this works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GGML_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # the ggml/ tree
GGML_DIR="${GGML_ROOT}/vendor/ggml"
PATCH_DIR="${GGML_ROOT}/vendor/ggml-patches"

if [[ ! -d "${GGML_DIR}" ]]; then
    echo "error: ggml submodule not found at ${GGML_DIR}" >&2
    echo "       did you forget 'git submodule update --init --recursive'?" >&2
    exit 1
fi

if [[ ! -d "${GGML_DIR}/.git" && ! -f "${GGML_DIR}/.git" ]]; then
    echo "error: ${GGML_DIR} is not a git repository" >&2
    exit 1
fi

if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "error: patch directory not found at ${PATCH_DIR}" >&2
    exit 1
fi

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "ggml patches: no patches found in ${PATCH_DIR} (nothing to do)"
    exit 0
fi

# Sort by filename so the numeric prefix (0001-, 0002-, ...) determines order.
IFS=$'\n' PATCHES=($(printf '%s\n' "${PATCHES[@]}" | sort))
unset IFS

applied=0
skipped=0

cd "${GGML_DIR}"

for patch in "${PATCHES[@]}"; do
    name="$(basename "${patch}")"

    # Already applied? `git apply --check --reverse` succeeds iff every hunk
    # is currently present in the tree (i.e. we *could* roll it back).
    if git apply --check --reverse "${patch}" >/dev/null 2>&1; then
        echo "ggml patches: skipping ${name} (already applied)"
        skipped=$((skipped + 1))
        continue
    fi

    # Otherwise it must apply cleanly forward.
    if git apply --check "${patch}" >/dev/null 2>&1; then
        if ! git apply "${patch}"; then
            echo "error: failed to apply ${name} after --check succeeded" >&2
            echo "       this should not happen; the submodule tree may be dirty" >&2
            exit 1
        fi
        echo "ggml patches: applied ${name}"
        applied=$((applied + 1))
        continue
    fi

    # Neither forward-applicable nor already-applied: bail with diagnostics.
    echo "error: cannot apply ${name}" >&2
    echo "       'git apply --check' output (forward):" >&2
    git apply --check "${patch}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       'git apply --check --reverse' output:" >&2
    git apply --check --reverse "${patch}" 2>&1 | sed 's/^/         /' >&2 || true
    echo "       submodule HEAD: $(git rev-parse HEAD)" >&2
    echo "       try: cd ${GGML_DIR} && git status" >&2
    exit 1
done

echo "ggml patches: applied ${applied}, skipped ${skipped}"
