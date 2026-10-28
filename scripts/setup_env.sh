#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${LLACO_ENV:-${VIRTUAL_ENV:-$ROOT/.venv}}"
PYTHON="$ENV_DIR/bin/python"
[[ -x "$PYTHON" ]] || {
  echo 'Create and activate a Python 3.12 environment as described in README.md.' >&2
  exit 2
}
cd "$ROOT"
"$PYTHON" -c 'import torch' || {
  echo 'Install PyTorch first using the command in README.md.' >&2
  exit 2
}
"$PYTHON" -m pip install -r requirements.txt
SITE_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$ROOT" "$ROOT/Megatron-Bridge/src" "$ROOT/Megatron-Bridge/3rdparty/Megatron-LM" > "$SITE_DIR/llaco_sources.pth"
"$PYTHON" -m pip check
