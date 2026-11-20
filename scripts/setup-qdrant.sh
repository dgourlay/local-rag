#!/usr/bin/env bash
set -euo pipefail

docker pull qdrant/qdrant:v1.20
# Publish on loopback only: a bare "-p 6333:6333" listens on every interface,
# exposing the unauthenticated index to the network. 6334 is gRPC, which the
# client uses by default (qdrant.prefer_grpc).
docker run -d \
  --name local-rag-qdrant \
  -p 127.0.0.1:6333:6333 \
  -p 127.0.0.1:6334:6334 \
  -v local-rag-qdrant-data:/qdrant/storage \
  qdrant/qdrant:v1.20

echo "Qdrant running at http://127.0.0.1:6333 (gRPC 127.0.0.1:6334)"
