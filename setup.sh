#!/usr/bin/env bash
# One-time setup for AutoLabel-Tool (macOS / Linux).
#   ./setup.sh
# Options (environment variables):
#   PYTHON=/path/to/python3.12   Python used to build the environments (3.10–3.12)
#   SAM3_PYTHON=/path/to/python  reuse an existing SAM3 environment
#   TORCH_INDEX_URL=...          PyTorch wheel index for SAM3 (Linux + CUDA, e.g. https://download.pytorch.org/whl/cu128)
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
SAM3_COMMIT="f92a5ef"   # facebookresearch/sam3 commit this tool is tested with

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for c in python3.12 python3.11 python3.10 \
           "$HOME"/.local/share/uv/python/cpython-3.12*/bin/python3.12 \
           /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then
      v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
      case "$v" in 3.10|3.11|3.12) echo "$c"; return;; esac
    fi
  done
}
PY="$(pick_python)"
[ -n "$PY" ] || die "Python 3.10–3.12 not found. Install one (e.g. 'brew install python@3.12' or 'uv python install 3.12'), or set PYTHON=..."
say "Using $PY ($("$PY" --version))"

# ---------------------------------------------------------------- agent environment
say "Creating the agent environment (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
chmod +x autolabel.sh

# ---------------------------------------------------------------- SAM3 environment
sam3_ok() { "$1" -c "import sam3, torch, cv2, PIL; from sam3.model_builder import build_sam3_image_model" >/dev/null 2>&1; }
SAM3_PY="${SAM3_PYTHON:-}"
if [ -z "$SAM3_PY" ] && [ -x "$HOME/sam3/.venv/bin/python" ] && sam3_ok "$HOME/sam3/.venv/bin/python"; then
  SAM3_PY="$HOME/sam3/.venv/bin/python"
  say "Found an existing SAM3 install: $SAM3_PY"
fi
if [ -z "$SAM3_PY" ]; then
  say "Installing SAM3 into third_party/sam3 (PyTorch download is large)"
  command -v git >/dev/null || die "git is required to download SAM3"
  mkdir -p third_party
  [ -d third_party/sam3/.git ] || git clone -q --filter=blob:none https://github.com/facebookresearch/sam3.git third_party/sam3
  # Pinned commit + patch: picks cuda / mps / cpu and avoids the CUDA-only Triton import,
  # so SAM3 also runs on Apple Silicon (MPS) and CPU.
  git -C third_party/sam3 checkout -q "$SAM3_COMMIT"
  if git -C third_party/sam3 apply --check "$HERE/patches/sam3_mac_mps.patch" 2>/dev/null; then
    git -C third_party/sam3 apply "$HERE/patches/sam3_mac_mps.patch"
  fi
  [ -x third_party/sam3/.venv/bin/python ] || "$PY" -m venv third_party/sam3/.venv
  SAM3_PY="$HERE/third_party/sam3/.venv/bin/python"
  "$SAM3_PY" -m pip install -q --upgrade pip
  if [ -n "${TORCH_INDEX_URL:-}" ]; then
    "$SAM3_PY" -m pip install -q torch torchvision --index-url "$TORCH_INDEX_URL"
  else
    "$SAM3_PY" -m pip install -q torch torchvision
  fi
  # SAM3 imports pkg_resources (removed in setuptools 81) and psutil
  "$SAM3_PY" -m pip install -q -e third_party/sam3 "setuptools<81" psutil opencv-python pillow einops
fi
sam3_ok "$SAM3_PY" || die "SAM3 environment at $SAM3_PY is not usable (import failed)"
printf 'SAM3_PYTHON=%q\n' "$SAM3_PY" > config.env
say "SAM3 python: $SAM3_PY (saved to config.env)"

say "Checking access to the SAM3 checkpoint on Hugging Face (gated)"
if "$SAM3_PY" -c "from huggingface_hub import hf_hub_download as d; d('facebook/sam3', 'config.json')" >/dev/null 2>&1; then
  echo "OK: reachable (~3.4 GB download on the first start)."
else
  HF_CLI="$(dirname "$SAM3_PY")/hf"
  cat <<EOF
The SAM3 checkpoint is gated. Before starting:
  1. Request access at https://huggingface.co/facebook/sam3 and wait for approval.
  2. Create a token at https://huggingface.co/settings/tokens
  3. Log in on this machine:   "$HF_CLI" auth login
EOF
fi

say "Done. Next:"
cat <<'EOF'
  ./autolabel.sh login --server http://localhost:8080   # paste a CVAT Personal Access Token
  ./autolabel.sh register                               # creates the functions in CVAT
  ./autolabel.sh start                                  # or: ./autolabel.sh install-service
EOF
