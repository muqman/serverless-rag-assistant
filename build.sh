#!/bin/bash
# Packages each Lambda into its own zip. Run from the project root.
set -e
cd "$(dirname "$0")"
rm -rf build && mkdir -p build/ingest build/api
python3 -m pip install pypdf --target build/ingest --quiet
cp src/rag_core.py src/ingest_handler.py build/ingest/
cp src/rag_core.py src/api_handler.py build/api/
(cd build/ingest && zip -qr ../ingest.zip .)
(cd build/api && zip -qr ../api.zip .)
ls -lh build/*.zip
