#!/usr/bin/env bash
set -euo pipefail

EXTRAS="${TRACERIGOR_EXTRAS:-analysis,data,envs,judge}"

python -m pip install --upgrade pip
python -m pip install -e ".[${EXTRAS}]"

echo "TraceRigor installed with extras: ${EXTRAS}"
echo "Training additionally requires a compatible VERL runtime and GPU stack."
