#!/usr/bin/env bash
# One background job per experiment; unique TAG, logs under results/.
# TAG=u2 PROFILE=u2 nohup ./scripts/run_config.sh > results/run-u2.log 2>&1 &
# Owns restart, watchdog, suites and graceful shutdown. Never foreground it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
load_hf_token
export IMAGE CONTAINER REVISION SERVED_NAME HF_CACHE PLE_DIR SGLANG_CACHE BUILD SNAPSHOT
export RECIPE_MODEL="${MODEL}"
cd "${ROOT}"
exec python3 -u "${SCRIPT_DIR}/experiment.py" run
