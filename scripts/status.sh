#!/usr/bin/env bash
# One-command orientation: what has run, what grokked, what is still going.
#   ./scripts/status.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "=== workers and run counts ==="
./scripts/remote/gtda-remote status 2>/dev/null | grep -E "running|finished|started of|FAILURES" || true

echo
echo "=== grokking step by run set ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR imperial \
    "cd /vol/bitbucket/oc525/grokking-tda && python3 scripts/remote/run_table.py" 2>/dev/null

echo
echo "=== analysis coverage ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o LogLevel=ERROR imperial \
    "cd /vol/bitbucket/oc525/grokking-tda && \
     echo \"analysed \$(ls -d results/raw/*/analysis 2>/dev/null | wc -l) of \$(ls -d results/raw/*/ | wc -l) runs\"; \
     head -5 results/logs/analysis_failures.txt 2>/dev/null || true" 2>/dev/null
