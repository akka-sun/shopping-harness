#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/base/Qwen3.5-2B}"
NETWORK_TURBO_FILE="${NETWORK_TURBO_FILE:-/etc/network_turbo}"
HF_CLI="${HF_CLI:-$ROOT/.venv/bin/hf}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

if [[ ! -x "$HF_CLI" ]]; then
  echo "Hugging Face CLI not found: $HF_CLI; run bash scripts/setup.sh" >&2
  exit 1
fi

if [[ -f "$NETWORK_TURBO_FILE" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  # AutoDL provides this shell fragment for Hugging Face/GitHub acceleration.
  # shellcheck source=/dev/null
  source "$NETWORK_TURBO_FILE" >/dev/null
fi

# The current container's Xet transport can wait indefinitely behind its proxy.
# Plain HTTP downloads retain Hugging Face Hub's resumable cache semantics.
export HF_HUB_DISABLE_XET=1

mkdir -p "$OUTPUT_DIR"
echo "downloading model=$MODEL"
echo "destination=$OUTPUT_DIR"
"$HF_CLI" download "$MODEL" \
  --local-dir "$OUTPUT_DIR" \
  --max-workers "$HF_MAX_WORKERS"

"$ROOT/.venv/bin/python" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = root / "config.json"
if not config.is_file():
    raise SystemExit(f"download verification failed: missing {config}")

weights = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
if not weights:
    raise SystemExit(f"download verification failed: no model weight files in {root}")

for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    index_path = root / index_name
    if not index_path.is_file():
        continue
    index = json.loads(index_path.read_text(encoding="utf-8"))
    referenced = {str(name) for name in (index.get("weight_map") or {}).values()}
    missing = sorted(name for name in referenced if not (root / name).is_file())
    if missing:
        raise SystemExit(
            f"download verification failed: missing {len(missing)} referenced shards: "
            + ", ".join(missing[:5])
        )

total_bytes = sum(path.stat().st_size for path in weights)
print(
    f"model download verified: {root} "
    f"weight_files={len(weights)} weight_bytes={total_bytes}"
)
PY
