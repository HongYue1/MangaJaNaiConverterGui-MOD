#!/usr/bin/env bash
# JaNai Upscaler - Linux setup.
#
# Builds a self-contained environment inside this folder, the same way
# setup.ps1 does on Windows:
#
#     backend/python        the environment the app runs in
#     backend/pythons       the managed CPython the environment is built on
#     backend/models        model weights
#     backend/src           upscaling backend, part of this repository
#     backend/_cache        uv cache and downloads - safe to delete
#     janai.config.json     what was resolved, and how
#     janai.runtime.txt     interpreter path, for the launcher
#
# Usage:
#     ./setup.sh [--python 3.13] [--torch auto|cpu|cu126|cu128|cu129]
#                [--models all|manga|illustration|none]
#                [--force] [--no-jxl-plugin] [--offline]
#
# Nothing is installed system-wide and nothing outside this folder is touched.

set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BACKEND="$HERE/backend"
PY_ROOT="$BACKEND/python"
MODELS_DIR="$BACKEND/models"
CACHE="$BACKEND/_cache"
TOOLS_DIR="$BACKEND/tools"
WORKER="$HERE/src/janai/worker/worker.py"
RESOLVER="$HERE/src/janai/core/paths.py"
REQUIREMENTS="$HERE/requirements.txt"
CONFIG_FILE="$HERE/janai.config.json"
RUNTIME_FILE="$HERE/janai.runtime.txt"
LAUNCHER="$HERE/janai-upscaler.sh"

PYTHON_VERSION=3.13
TORCH=auto
MODELS=all
FORCE=0
OFFLINE=0
JXL_PLUGIN=1

# Keep uv's state inside the folder: the managed interpreters and the wheel
# cache both live under backend/, so nothing is left behind in ~/.cache.
export UV_CACHE_DIR="$CACHE/uv"
export UV_PYTHON_INSTALL_DIR="$BACKEND/pythons"
export UV_NO_CONFIG=1

MODEL_PACKS="
manga|https://github.com/the-database/mangajanai/releases/download/1.0.0/MangaJaNai_V1_ModelsOnly.zip
illustration|https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3denoise.zip
illustration|https://github.com/the-database/MangaJaNai/releases/download/3.0.0/IllustrationJaNai_V3detail.zip
"

# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
if [ -t 1 ]; then
    C_STEP=$'\033[1m'; C_DIM=$'\033[90m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
    C_STEP=''; C_DIM=''; C_WARN=''; C_ERR=''; C_OFF=''
fi

step() { printf '\n%s==> %s%s\n' "$C_STEP" "$1" "$C_OFF"; }
info() { printf '    %s%s%s\n' "$C_DIM" "$1" "$C_OFF"; }
warn() { printf '    %swarning: %s%s\n' "$C_WARN" "$1" "$C_OFF" >&2; }
die() { printf '\n    %serror: %s%s\n\n' "$C_ERR" "$1" "$C_OFF" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
while [ $# -gt 0 ]; do
    case "$1" in
        --python) PYTHON_VERSION=${2:-} ; shift 2 ;;
        --torch) TORCH=${2:-} ; shift 2 ;;
        --models) MODELS=${2:-} ; shift 2 ;;
        --force) FORCE=1 ; shift ;;
        --offline) OFFLINE=1 ; shift ;;
        --no-jxl-plugin) JXL_PLUGIN=0 ; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^#\{1,\} \{0,1\}//' ; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

case "$TORCH" in
    auto|cpu|cu126|cu128|cu129) ;;
    *) die '--torch must be auto, cpu, cu126, cu128 or cu129' ;;
esac
case "$MODELS" in
    all|manga|illustration|none) ;;
    *) die '--models must be all, manga, illustration or none' ;;
esac

printf '\n  %sJaNai Upscaler - setup%s\n' "$C_STEP" "$C_OFF"
info "app folder: $HERE"

# --------------------------------------------------------------------------- #
step 'Checking the host'
[ "$(uname -s)" = Linux ] || warn "this script targets Linux; $(uname -s) is untested"
for needed in "$WORKER" "$RESOLVER" "$REQUIREMENTS"; do
    [ -f "$needed" ] || die "$(basename "$needed") is missing - this checkout is incomplete"
done
info "$(uname -sr)  $(uname -m)"
mkdir -p "$BACKEND" "$CACHE" "$TOOLS_DIR" || die "cannot write inside $HERE"

# --------------------------------------------------------------------------- #
step 'Backend source and ICC profiles'
# src, ImageMagick and resources are tracked in this repository, so a clone
# already has them: nothing is copied and nothing is borrowed from elsewhere.
[ -f "$BACKEND/src/progress_controller.py" ] \
    || die 'backend/src is missing - restore it with: git checkout -- backend'
[ -f "$BACKEND/ImageMagick/Dot Gain 20%.icc" ] \
    || warn 'the ICC profiles are missing from backend/ImageMagick'
[ -d "$BACKEND/resources" ] || warn 'backend/resources is missing'
info 'present in this repository'

# --------------------------------------------------------------------------- #
step 'GPU'
if have nvidia-smi; then
    card=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    if [ -n "$card" ]; then info "$card"; else warn 'nvidia-smi did not report a GPU'; fi
else
    info 'nvidia-smi was not found'
    if [ "$TORCH" = auto ]; then
        warn 'no NVIDIA driver is visible - installing the CPU build of torch'
        TORCH=cpu
    fi
fi

# --------------------------------------------------------------------------- #
step 'uv'
UV=""
if [ -x "$TOOLS_DIR/uv" ]; then
    UV="$TOOLS_DIR/uv"
elif have uv; then
    UV=$(command -v uv)
elif [ "$OFFLINE" = 0 ] && have curl; then
    info 'downloading uv into backend/tools'
    UV_UNMANAGED_INSTALL="$TOOLS_DIR" INSTALLER_NO_MODIFY_PATH=1 \
        sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' >/dev/null 2>&1
    [ -x "$TOOLS_DIR/uv" ] && UV="$TOOLS_DIR/uv"
fi
if [ -n "$UV" ]; then
    info "$("$UV" --version 2>/dev/null | head -n 1)"
    info "$UV"
    info "cache      $UV_CACHE_DIR"
else
    warn 'uv is not available - falling back to python3 -m venv and pip'
    have python3 || die 'neither uv nor python3 is available; install Python 3.11+ and rerun'
fi

# --------------------------------------------------------------------------- #
step "Python $PYTHON_VERSION environment"
find_interpreter() {
    for candidate in "$PY_ROOT/bin/python3" "$PY_ROOT/bin/python"; do
        if [ -x "$candidate" ]; then printf '%s' "$candidate"; return 0; fi
    done
    return 1
}

# A folder left as a symlink by an older install is unlinked, never installed
# into: the runtime and the weights belong to this folder.
for leftover in "$PY_ROOT" "$MODELS_DIR"; do
    if [ -L "$leftover" ]; then
        rm -f "$leftover" && info "removed a leftover link at ${leftover#"$HERE"/}"
    fi
done

PY=$(find_interpreter) || PY=""
if [ -n "$PY" ] && [ "$FORCE" = 0 ]; then
    info 'backend/python already exists (use --force to rebuild it)'
else
    [ "$FORCE" = 1 ] && [ -d "$PY_ROOT" ] && rm -rf "$PY_ROOT"
    if [ -n "$UV" ]; then
        # --managed-python keeps the interpreter inside this folder rather than
        # binding the environment to whatever Python the distro ships.
        "$UV" venv "$PY_ROOT" --python "$PYTHON_VERSION" --managed-python \
            || { warn 'retrying without --managed-python'
                 "$UV" venv "$PY_ROOT" --python "$PYTHON_VERSION" || die 'uv venv failed'; }
    else
        python3 -m venv "$PY_ROOT" || die 'python3 -m venv failed (install the python3-venv package)'
    fi
    PY=$(find_interpreter) || die 'the new environment has no bin/python'
fi
info "$PY"
info "$("$PY" --version 2>&1 | head -n 1)"

# --------------------------------------------------------------------------- #
step 'Dependencies'
info "torch build: $TORCH"
case "$TORCH" in
    cpu) INDEX='https://download.pytorch.org/whl/cpu' ;;
    auto) INDEX='https://download.pytorch.org/whl/cu128' ;;
    *) INDEX="https://download.pytorch.org/whl/$TORCH" ;;
esac

installed=0
if [ "$OFFLINE" = 1 ]; then
    info 'skipped (--offline)'
    installed=1
elif [ -n "$UV" ]; then
    if "$UV" pip install --python "$PY" --torch-backend "$TORCH" -r "$REQUIREMENTS"; then
        installed=1
    else
        warn '--torch-backend was rejected; falling back to an explicit PyTorch index'
        "$UV" pip install --python "$PY" --extra-index-url "$INDEX" -r "$REQUIREMENTS" && installed=1
    fi
else
    "$PY" -m pip install --upgrade pip >/dev/null 2>&1
    "$PY" -m pip install --extra-index-url "$INDEX" -r "$REQUIREMENTS" && installed=1
fi
[ "$installed" = 1 ] || die 'installing the requirements failed - see the output above'

# --------------------------------------------------------------------------- #
step 'Models'
mkdir -p "$MODELS_DIR"
count_models() {
    find "$MODELS_DIR" -maxdepth 1 -type f \
        \( -name '*.pth' -o -name '*.safetensors' -o -name '*.onnx' \) 2>/dev/null | wc -l | tr -d ' '
}
fetch() {
    if have curl; then curl -fsSL "$1" -o "$2"
    elif have wget; then wget -q "$1" -O "$2"
    else return 1
    fi
}

if [ "$MODELS" = none ] || [ "$OFFLINE" = 1 ]; then
    info 'skipped'
else
    printf '%s\n' "$MODEL_PACKS" | while IFS='|' read -r key url; do
        [ -n "${url:-}" ] || continue
        [ "$MODELS" = all ] || [ "$MODELS" = "$key" ] || continue
        name=$(basename "$url")
        zip="$CACHE/$name"
        if [ ! -f "$zip" ]; then
            info "downloading $name"
            fetch "$url" "$zip" || { warn "could not download $name"; rm -f "$zip"; continue; }
        fi
        unpack=$(mktemp -d "$CACHE/unpack.XXXXXX") || continue
        # The stdlib unpacks it, so unzip is not a requirement.
        if "$PY" -m zipfile -e "$zip" "$unpack" >/dev/null 2>&1; then
            find "$unpack" -type f \( -name '*.pth' -o -name '*.safetensors' -o -name '*.onnx' \) \
                -exec cp -f {} "$MODELS_DIR/" \;
            info "unpacked $name"
        else
            warn "could not unpack $name"
        fi
        rm -rf "$unpack"
    done
fi
info "$(count_models) weight file(s) in backend/models"

# --------------------------------------------------------------------------- #
step 'Imaging'
if "$PY" -c 'import pyvips' >/dev/null 2>&1; then
    info 'pyvips loaded libvips'
else
    warn 'pyvips could not load libvips - install it from your package manager'
    info 'debian/ubuntu: sudo apt install libvips42   ·   fedora: sudo dnf install vips'
fi
if have cjxl; then
    info "cjxl: $(command -v cjxl)"
else
    info 'cjxl was not found; install libjxl-tools (apt) or libjxl (dnf) for the fastest JXL path'
fi
if [ "$JXL_PLUGIN" = 1 ] && [ "$OFFLINE" = 0 ]; then
    if [ -n "$UV" ]; then
        "$UV" pip install --python "$PY" pillow-jxl-plugin >/dev/null 2>&1 \
            && info 'installed pillow-jxl-plugin' \
            || warn 'pillow-jxl-plugin could not be installed; JXL may be unavailable'
    else
        "$PY" -m pip install -q pillow-jxl-plugin >/dev/null 2>&1 \
            && info 'installed pillow-jxl-plugin' \
            || warn 'pillow-jxl-plugin could not be installed; JXL may be unavailable'
    fi
fi

# --------------------------------------------------------------------------- #
step 'Configuration'
pyver=$("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null | head -n 1)
cat > "$CONFIG_FILE" <<EOF
{
  "python": "$pyver",
  "python_dir": "$PY_ROOT",
  "models_dir": "$MODELS_DIR",
  "torch_backend": "$TORCH",
  "platform": "linux",
  "written": "$(date +%Y-%m-%dT%H:%M:%S)"
}
EOF
info "wrote $(basename "$CONFIG_FILE")"
printf '%s\n' "$PY" > "$RUNTIME_FILE"
info "wrote $(basename "$RUNTIME_FILE") -> $PY"
[ -f "$LAUNCHER" ] && [ ! -x "$LAUNCHER" ] && chmod +x "$LAUNCHER" 2>/dev/null

# --------------------------------------------------------------------------- #
step 'What the app resolves'
"$PY" "$RESOLVER"

# --------------------------------------------------------------------------- #
step 'Self-test'
probe=$("$PY" "$WORKER" --probe 2>&1 | grep '"probe"' | head -n 1)
if [ -n "$probe" ]; then
    "$PY" -c '
import json, sys
raw = json.loads(sys.argv[1])
p = raw.get("probe") if isinstance(raw.get("probe"), dict) else raw
cuda = p.get("cuda")
print("    python {}  torch {}  {}".format(
    p.get("python", "?"), p.get("torch", "?"), "CUDA " + str(cuda) if cuda else "CPU build"))
gpus = p.get("gpus") or []
if isinstance(gpus, list) and gpus:
    names = [g.get("name", "?") if isinstance(g, dict) else str(g) for g in gpus]
    print("    gpu: " + ", ".join(names))
models = p.get("models") or []
print("    models: {}".format(len(models) if isinstance(models, (list, tuple)) else 0))
if not p.get("cuda"):
    print("    (no CUDA: upscaling will run on the CPU and be slow)")
' "$probe" 2>/dev/null || info "$probe"
else
    warn 'the worker did not report a probe - see the output above'
fi

# --------------------------------------------------------------------------- #
step 'Ready'
info 'start the app with: ./janai-upscaler.sh'
printf '\n'
