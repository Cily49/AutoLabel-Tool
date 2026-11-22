#!/usr/bin/env bash
# Entry point: ./autolabel.sh <command>   (see ./autolabel.sh -h)
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m autolabel.cli "$@"
