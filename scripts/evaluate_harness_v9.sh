#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/sft-curriculum-stage-c-harness-v9-active}"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Warm the embedded ShopSimulator and clear stale leases before collection.
# The service may briefly return 502 while its worker pool is initializing.
for attempt in $(seq 1 12); do
  curl -fsS -X POST "${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}/api/shop_agent" \
    -H 'Content-Type: application/json' \
    -d '{"action":"release_all"}' >/dev/null 2>&1 || true
  probe="$(curl -sS -o /tmp/shopping_grpo_probe.json -w '%{http_code}' \
    -X POST "${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}/api/shop_agent" \
    -H 'Content-Type: application/json' -d '{"action":"reset","idx":8187}' || true)"
  if [[ "$probe" == "200" ]] && grep -q '"env_idx"' /tmp/shopping_grpo_probe.json; then
    env_idx="$(sed -n 's/.*"env_idx":\([0-9][0-9]*\).*/\1/p' /tmp/shopping_grpo_probe.json | head -1)"
    curl -fsS -X POST "${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}/api/shop_agent" \
      -H 'Content-Type: application/json' -d "{\"action\":\"release_one\",\"env_idx\":${env_idx:-0}}" >/dev/null || true
    break
  fi
  sleep "$attempt"
done

"$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}" \
  --model "${SERVED_MODEL_NAME:-shopping-agent}" \
  --llm-base-url "${LLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --api-key "${LLM_API_KEY:-EMPTY}" \
  --infrastructure-retries "${INFRASTRUCTURE_RETRIES:-3}" \
  --context-compaction \
  --harness-memories outputs/harness/sft-curriculum-stage-c/regenerated-v6/memory/active.jsonl \
  --harness-skills outputs/harness/sft-curriculum-stage-c/regenerated-v6/skill/active.jsonl \
  --harness-source-trajectories outputs/harness/sft-curriculum-stage-c/regenerated-v6/probe/normalized-sample-200.jsonl \
  --activate-candidate-skills

"$ROOT/.venv/bin/python" - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
successes = summary.get("strict_successes", 0)
expected = summary.get("expected_tasks", 200)
print(json.dumps({"strict_successes": successes, "expected_tasks": expected,
                  "strict_success_rate": successes / expected if expected else 0.0},
                 ensure_ascii=False))
if expected == 200 and successes < 130:
    raise SystemExit("v9 Harness is below the required 65% strict-success target")
PY
