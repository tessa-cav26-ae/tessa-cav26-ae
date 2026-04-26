#!/usr/bin/env bash
# Reproduce loss.csv, loss.png, and landscape.png next to this script.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

python "$HERE/kydice.py" --model "$HERE/kydice.jani" --output-dir "$HERE" "$@"
