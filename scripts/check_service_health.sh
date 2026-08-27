#!/usr/bin/env bash
set -euo pipefail

ENV_URL="${ENV_URL:-}"
JUDGE_URL="${JUDGE_URL:-}"
MAX_WAIT="${MAX_WAIT:-120}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

if [ -z "$ENV_URL" ]; then
  echo "ERROR: ENV_URL must be set"
  exit 1
fi

wait_for_url() {
  local url="$1"
  local label="$2"
  local waited=0

  echo "Waiting for $label at $url ..."
  while ! curl -sf "$url" >/dev/null 2>&1; do
    sleep "$SLEEP_SECONDS"
    waited=$((waited + SLEEP_SECONDS))
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      echo "ERROR: $label did not become ready within ${MAX_WAIT}s"
      return 1
    fi
  done

  echo "$label ready after ${waited}s"
}

wait_for_url "$ENV_URL" "environment server"

if [ -n "$JUDGE_URL" ]; then
  wait_for_url "$JUDGE_URL" "judge server"
fi

echo "All requested services are healthy."