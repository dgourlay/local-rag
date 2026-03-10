#!/usr/bin/env bash
#
# Download and export the BGE reranker model to ONNX format.
# Idempotent -- skips export if the model already exists.
#
# Note: The reranker also auto-exports on first search if optimum is installed.
# This script is for pre-downloading so the first search is fast.

set -euo pipefail

# Use venv python if available, otherwise system python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    PIP="${PROJECT_DIR}/.venv/bin/pip"
else
    PYTHON="python3"
    PIP="pip3"
fi

MODEL_DIR="${HOME}/.cache/dropbox-rag/models/bge-reranker-v2-m3"
MODEL_FILE="${MODEL_DIR}/model.onnx"

echo "=== dropbox-rag: Reranker Model Export ==="
echo ""

if [ -f "${MODEL_FILE}" ]; then
    echo "Reranker ONNX model already exists at:"
    echo "  ${MODEL_FILE}"
    echo ""
    echo "Skipping export. To force re-export, delete the directory:"
    echo "  rm -rf ${MODEL_DIR}"
    exit 0
fi

echo "Reranker ONNX model not found. Exporting from HuggingFace..."
echo ""

# Check if optimum is available; install if not
if ! "${PYTHON}" -c "import optimum" 2>/dev/null; then
    echo "Installing optimum[onnxruntime] (required for ONNX export)..."
    "${PIP}" install "optimum[onnxruntime]"
    echo ""
fi

echo "Exporting BAAI/bge-reranker-v2-m3 to ONNX format..."
echo "This may take a few minutes on first run."
echo ""

"${PYTHON}" -c "
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import logging

logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

print('Downloading and converting model...')
model = ORTModelForSequenceClassification.from_pretrained(
    'BAAI/bge-reranker-v2-m3', export=True
)

print('Saving ONNX model to ${MODEL_DIR}/')
model.save_pretrained('${MODEL_DIR}')

print('Saving tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
tokenizer.save_pretrained('${MODEL_DIR}')

print('Done.')
"

echo ""
echo "Reranker model exported successfully to:"
echo "  ${MODEL_DIR}/"
