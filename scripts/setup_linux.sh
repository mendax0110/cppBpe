#!/usr/bin/env bash
set -euo pipefail

packages=(
  build-essential
  cmake
  libtbb-dev
  python3-dev
  python3-venv
  python3-pip
)

missing=()
for pkg in "${packages[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    missing+=("$pkg")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  sudo apt-get update -qq
  sudo apt-get install -y "${missing[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python python3 --clear
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
