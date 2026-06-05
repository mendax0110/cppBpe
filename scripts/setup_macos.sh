#!/usr/bin/env bash
set -euo pipefail

brew install cmake tbb python@3.11 uv

uv venv --python 3.11 --clear
source .venv/bin/activate

uv sync
uv pip install nanobind

PYTHON="$(pwd)/.venv/bin/python"
VENV_PREFIX="$(pwd)/.venv"

NANOBIND_DIR="$($PYTHON -c 'import nanobind; print(nanobind.cmake_dir())')"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCPPBPE_BUILD_TESTS=ON \
    -DCPPBPE_USE_BUNDLED_PCRE2=ON \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DCMAKE_PREFIX_PATH="$VENV_PREFIX" \
    -Dnanobind_DIR="$NANOBIND_DIR"

cmake --build build --parallel "$(sysctl -n hw.logicalcpu)"

./build/cppBpe_tests

PYTHONPATH=build "$PYTHON" -m pytest tests/python/test_tokenizer.py -v

PYTHONPATH=build "$PYTHON" tests/python/benchmark.py --vocab 8192 --n 100 --probe 100000 --corpus input.txt