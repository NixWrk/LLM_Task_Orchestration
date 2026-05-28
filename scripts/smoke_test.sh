#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dotenv_get() {
  local key="$1"
  local env_file="$ROOT_DIR/.env"

  if [[ ! -f "$env_file" ]]; then
    return 0
  fi

  local line
  line="$(grep -E "^${key}=" "$env_file" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    printf '%s' "${line#*=}"
  fi
}

BASE_URL="${ORCHESTRATOR_BASE_URL_PUBLIC:-$(dotenv_get ORCHESTRATOR_BASE_URL_PUBLIC)}"
BASE_URL="${BASE_URL:-http://localhost:4100}"

API_KEY="${LITELLM_MASTER_KEY:-$(dotenv_get LITELLM_MASTER_KEY)}"
API_KEY="${API_KEY:-sk-change-this-local-key}"

MODEL="${PUBLIC_MODEL_NAME:-$(dotenv_get PUBLIC_MODEL_NAME)}"
MODEL="${MODEL:-local-main}"

PAYLOAD="$(printf '{"model":"%s","messages":[{"role":"user","content":"Return exactly: ok"}],"temperature":0,"max_tokens":8}' "$MODEL")"

curl --fail --silent --show-error \
  "${BASE_URL%/}/v1/chat/completions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

printf '\nSmoke test passed\n'
