#!/usr/bin/env bash
# ==============================================================================
# 1-Click NVIDIA GPU Driver & CUDA ONNXRuntime Setup for ISDC-27 FPV ML Pipeline
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================================================="
echo "  ISDC-27: Enabling NVIDIA GPU (PCI 01:00.0) & CUDA Execution Provider"
echo "=============================================================================="

echo "[1/3] Installing Ubuntu recommended NVIDIA proprietary driver (requires sudo password)..."
sudo ubuntu-drivers install

echo "[2/3] Upgrading virtual environment to onnxruntime-gpu..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade onnxruntime-gpu
"$SCRIPT_DIR/.venv/bin/pip" freeze > "$SCRIPT_DIR/requirements.txt"

echo "[3/3] NVIDIA driver & onnxruntime-gpu installed!"
echo "------------------------------------------------------------------------------"
echo "  IMPORTANT: Please reboot your laptop once ('sudo reboot') so the Linux"
echo "  kernel unloads 'nouveau' and loads the NVIDIA CUDA driver."
echo "  After rebooting, verify with: nvidia-smi"
echo "  pipeline.py will automatically switch to CUDAExecutionProvider!"
echo "=============================================================================="
