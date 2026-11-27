#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

uv run gosmart-clinical-nlu train \
  --training-dir "${REPO_ROOT}/training" \
  --out "${REPO_ROOT}/nlu_models/intent_models.zip"
