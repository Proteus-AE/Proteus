#!/usr/bin/env bash
# Fetch and build the external simulators used by the cross-validation
# experiments (ext/README.md). Idempotent; safe to re-run.
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

clone_pinned() {  # name url sha
    local dir="ext/$1"
    if [ -d "$dir/.git" ]; then
        echo "== $1: already cloned"
    else
        echo "== $1: cloning $2"
        git clone "$2" "$dir"
    fi
    git -C "$dir" fetch -q origin "$3" 2>/dev/null || true
    git -C "$dir" checkout -q "$3" || {
        echo "   pinned revision $3 not found; staying on default branch"; }
}

while read -r name url sha; do
    case "$name" in \#*|"") continue;; esac
    clone_pinned "$name" "$url" "$sha"
done < ext/VERSIONS

# ---- ramulator2: stock build (host-path cross-check) -----------------------
if [ -d ext/ramulator2 ]; then
    echo "== ramulator2: building (stock; needs a C++20 compiler)"
    { cmake -S ext/ramulator2 -B ext/ramulator2/build \
            -DCMAKE_BUILD_TYPE=Release >/dev/null &&
      cmake --build ext/ramulator2/build -j"$(nproc 2>/dev/null || echo 4)"
    } || echo "   build failed -- check the compiler (g++-12+) and re-run"
fi

# ---- ramulator2-pim: patched tree (PIM device model) -----------------------
if [ -d ext/ramulator2 ] && [ ! -d ext/ramulator2-pim ]; then
    echo "== ramulator2-pim: applying integration/ramulator2/patches"
    cp -r ext/ramulator2 ext/ramulator2-pim
    rm -rf ext/ramulator2-pim/build
    (cd ext/ramulator2-pim &&
     git apply "$ROOT"/integration/ramulator2/patches/*.patch)
    echo "== ramulator2-pim: building"
    cmake -S ext/ramulator2-pim -B ext/ramulator2-pim/build \
          -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build ext/ramulator2-pim/build \
          -j"$(nproc 2>/dev/null || echo 4)" || {
        echo "   patched build failed (upstream API drift?); the stock tree"
        echo "   is unaffected and the host-path cross-check still works."; }
fi

# ---- ONNXim ---------------------------------------------------------------
if [ -d ext/onnxim ]; then
    echo "== onnxim: see ext/onnxim/README.md for its build (protobuf etc.);"
    echo "   experiments/run_onnxim_xcheck.py picks the binary up from"
    echo "   ext/onnxim/build or \$ONNXIM_HOME once built."
fi

echo "== done"
