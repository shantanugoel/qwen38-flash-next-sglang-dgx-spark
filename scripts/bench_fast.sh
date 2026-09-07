#!/usr/bin/env bash
# Benchmark-only mode for an existing server; same lock as run_config.sh.
# Background only. Every suite gets a bounded process, log and JSON verdict.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
exec python3 -u "${SCRIPT_DIR}/experiment.py" suite
