#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SUITE:?Set SUITE to your suite JSON path}"
exec "${PYTHON:-python}" -u -m osd_tracedraft.queue --config "$SUITE" --gpus "${GPUS:-0,1,2,3}" "$@"
