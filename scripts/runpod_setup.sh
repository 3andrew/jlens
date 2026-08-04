#!/usr/bin/env bash
# One-time bootstrap for a fresh RunPod pod (PyTorch template).
# Usage: bash scripts/runpod_setup.sh
set -euo pipefail

# HF cache on the persistent volume so model downloads survive pod restarts.
export HF_HOME=${HF_HOME:-/workspace/hf}
grep -q HF_HOME ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

pip install -e .

# Qwen3.5 fast-path kernels (hybrid linear-attention/conv blocks). If either
# build fails, skip — the torch fallback is correct, just slower.
pip install flash-linear-attention causal-conv1d || \
  echo "WARN: fast-path kernels failed to build; continuing with torch fallback"

python - << 'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.get_device_name(0))
EOF
