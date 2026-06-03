#!/usr/bin/env bash
set -euo pipefail

brew install cmake tbb pybind11 python@3.11 uv

uv venv --python 3.11 --clear
source .venv/bin/activate
uv sync

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCPPBPE_BUILD_TESTS=ON \
  -DCPPBPE_USE_BUNDLED_PCRE2=ON \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build build

./build/cppBpe_tests
PYTHONPATH=build ./.venv/bin/python -m pytest tests/python/test_tokenizer.py -v

PYTHONPATH=build ./.venv/bin/python tests/python/benchmark.py --vocab 8192 --n 100 --probe 100000 --corpus input.txt
