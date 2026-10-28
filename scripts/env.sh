#!/usr/bin/env bash
# Source from any directory to use this checkout and its Python environment.
LLACO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLACO_ROOT
export LLACO_ENV="${LLACO_ENV:-${VIRTUAL_ENV:-$LLACO_ROOT/.venv}}"
export PATH="$LLACO_ENV/bin:$PATH"
export PYTHON_BIN="$LLACO_ENV/bin/python"
export PYTHONPATH="$LLACO_ROOT:$LLACO_ROOT/Megatron-Bridge/src:$LLACO_ROOT/Megatron-Bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
