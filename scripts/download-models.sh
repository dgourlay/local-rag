#!/usr/bin/env bash
#
# Download embedding and reranker models for local-rag.
# Idempotent -- skips downloads if models already exist.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

CACHE_DIR="${HOME}/.cache/local-rag/models"

# --- 1. Embedding model (BAAI/bge-m3) ---
echo "=== local-rag: Embedding Model ==="
echo ""

"${PYTHON}" -c "
from sentence_transformers import SentenceTransformer
import os

cache_dir = '${CACHE_DIR}'
model_path = os.path.join(cache_dir, 'BAAI_bge-m3')
# sentence-transformers replaces / with _ in cache path
alt_path = os.path.join(cache_dir, 'models--BAAI--bge-m3')

if os.path.isdir(model_path) or os.path.isdir(alt_path):
    print('Embedding model already cached. Skipping.')
else:
    print('Downloading BAAI/bge-m3 embedding model...')
    print('This may take a few minutes (~2GB).')
    SentenceTransformer('BAAI/bge-m3', cache_folder=cache_dir)
    print('Embedding model downloaded.')
"

echo ""

# --- 2. Reranker model (BAAI/bge-reranker-v2-m3 → ONNX) ---
RERANKER_DIR="${CACHE_DIR}/bge-reranker-v2-m3"
RERANKER_FILE="${RERANKER_DIR}/model.onnx"

echo "=== local-rag: Reranker Model ==="
echo ""

if [ -f "${RERANKER_FILE}" ]; then
    echo "Reranker ONNX model already exists. Skipping."
else
    echo "Exporting BAAI/bge-reranker-v2-m3 to ONNX format..."
    echo "This may take a few minutes."
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

print('Saving ONNX model to ${RERANKER_DIR}/')
model.save_pretrained('${RERANKER_DIR}')

print('Saving tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
tokenizer.save_pretrained('${RERANKER_DIR}')

print('Done.')
"
fi

echo ""
echo "All models ready."
