#!/usr/bin/env sh
# JaNai Upscaler - Linux launcher. Uses the environment in ./backend/python.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
MAIN="$HERE/src/janai/app/main.py"
PY=""

# 1. whatever setup recorded
if [ -f "$HERE/janai.runtime.txt" ]; then
    PY=$(head -n 1 "$HERE/janai.runtime.txt" | tr -d '\r\n')
    [ -x "$PY" ] || PY=""
fi

# 2. the virtual environment setup.sh builds
if [ -z "$PY" ]; then
    for candidate in "$HERE/backend/python/bin/python3" "$HERE/backend/python/bin/python"; do
        if [ -x "$candidate" ]; then
            PY="$candidate"
            break
        fi
    done
fi

if [ -n "$PY" ]; then
    exec "$PY" "$MAIN" "$@"
fi

echo "No environment in backend/python - run ./setup.sh once to create it." >&2
echo "Trying the system Python: the interface will open, but upscaling needs setup." >&2
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$MAIN" "$@"
fi

echo "No Python was found at all. Run ./setup.sh first." >&2
exit 1
