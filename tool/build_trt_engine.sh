#!/bin/sh
# Resolve paths from this script, independent of the caller's working directory.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$SCRIPT_DIR/build_trt_engine.py" "$@"
