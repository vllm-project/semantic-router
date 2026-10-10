#!/usr/bin/env bash
# Idempotent environment for proxy builds inside the vLLM ROCm image (run from a CPU Job).
#   bash d25/omni/proxy/setup.sh [--probe]
# Python packages go to $PYLIB (pip --target), the Chromium build to $PLAYWRIGHT_BROWSERS_PATH, and the
# shared libraries Chromium needs to $D25_PROXY_SYSLIB (unpacked .deb files, nothing installed in the
# image). Source this file's exports with: eval "$(bash d25/omni/proxy/setup.sh --env)".
set -euo pipefail

OMNI=${OMNI:-/data/d25/omni}
PYLIB=${PYLIB:-$OMNI/pylib/proxy}
BROWSERS=${PLAYWRIGHT_BROWSERS_PATH:-$OMNI/pylib/proxy-browsers}
SYSLIB=${D25_PROXY_SYSLIB:-$OMNI/pylib/proxy-syslib}
PY_STAMP=v3
SYS_STAMP=v1
BROWSER_STAMP=v2
# --no-deps keeps the image's numpy, pillow and requests; listed here are the missing pure deps.
PACKAGES=(playwright==1.63.0 greenlet==3.5.6 pyee==13.0.1 remotezip==0.12.6 tabulate==0.9.0
  h5py==3.16.0 matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 fonttools==4.66.1
  kiwisolver==1.5.1 pyparsing==3.3.3)
CHROMIUM_DEBS=(libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libpangocairo-1.0-0
  libcairo2 libasound2 libatspi2.0-0 libxshmfence1 libfontconfig1 libfreetype6 fonts-liberation
  fonts-dejavu-core fonts-noto-core libx11-xcb1 libxcb-dri3-0 libdbus-1-3 libexpat1)

env_lines() {
  local lib="$SYSLIB/usr/lib/x86_64-linux-gnu:$SYSLIB/lib/x86_64-linux-gnu"
  echo "export PYTHONPATH=\"\${PYTHONPATH:-}:$PYLIB\""
  echo "export PLAYWRIGHT_BROWSERS_PATH=$BROWSERS"
  echo "export LD_LIBRARY_PATH=\"$lib:\${LD_LIBRARY_PATH:-}\""
  echo "export FONTCONFIG_FILE=$SYSLIB/fonts.conf"
}

if [[ "${1:-}" == "--env" ]]; then
  env_lines
  exit 0
fi

probe() {
  python -V
  head -2 /etc/os-release
  id -u
  python - <<'EOF'
import importlib
for name in ["numpy", "PIL", "matplotlib", "cv2", "scipy", "pandas", "pyarrow", "datasets",
             "huggingface_hub", "h5py", "torch", "transformers", "requests", "playwright", "remotezip"]:
    try:
        module = importlib.import_module(name)
        print(f"{name} {getattr(module, '__version__', '?')}")
    except Exception as error:
        print(f"{name} MISSING ({type(error).__name__})")
EOF
  echo "fonts: $(fc-list 2>/dev/null | wc -l)"
  df -h /tmp "$OMNI/proxy" | tail -2
}

if [[ "${1:-}" == "--probe" ]]; then
  probe
fi

mkdir -p "$PYLIB" "$BROWSERS" "$SYSLIB"
if [[ ! -f "$PYLIB/.stamp-$PY_STAMP" ]]; then
  echo "installing python overlay into $PYLIB"
  rm -rf "${PYLIB:?}"/*
  pip install -q --no-cache-dir --no-deps --target "$PYLIB" "${PACKAGES[@]}" > /tmp/pip.log 2>&1 \
    || { tail -20 /tmp/pip.log; exit 1; }
  touch "$PYLIB/.stamp-$PY_STAMP"
fi
export PYTHONPATH="${PYTHONPATH:-}:$PYLIB"

if [[ ! -f "$SYSLIB/.stamp-$SYS_STAMP" ]]; then
  work=$(mktemp -d)
  apt-get update -qq >/dev/null
  available=()
  for name in "${CHROMIUM_DEBS[@]}"; do
    if apt-cache show "${name}t64" >/dev/null 2>&1; then
      available+=("${name}t64")
    elif apt-cache show "$name" >/dev/null 2>&1; then
      available+=("$name")
    fi
  done
  needed=()
  for name in $(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts \
      --no-breaks --no-replaces --no-enhances "${available[@]}" | grep '^\w' | sort -u); do
    status=$(dpkg-query -W -f='${Status}' "$name" 2>/dev/null || true)
    [[ "$status" == "install ok installed" ]] || needed+=("$name")
  done
  echo "unpacking ${#needed[@]} packages into $SYSLIB"
  (cd "$work" && for name in "${needed[@]}"; do apt-get download -qq "$name" 2>/dev/null || true; done)
  for deb in "$work"/*.deb; do dpkg -x "$deb" "$SYSLIB"; done
  cat > "$SYSLIB/fonts.conf" <<EOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>$SYSLIB/usr/share/fonts</dir>
  <dir>$PYLIB/fonts</dir>
  <dir>/usr/share/fonts</dir>
  <cachedir>/tmp/fontconfig-cache</cachedir>
</fontconfig>
EOF
  rm -rf "$work"
  touch "$SYSLIB/.stamp-$SYS_STAMP"
fi
eval "$(env_lines)"

if [[ ! -f "$BROWSERS/.stamp-$BROWSER_STAMP" ]]; then
  echo "installing chromium into $BROWSERS"
  python -m playwright install chromium-headless-shell > /tmp/pw.log 2>&1 || { tail -20 /tmp/pw.log; exit 1; }
  touch "$BROWSERS/.stamp-$BROWSER_STAMP"
fi

if [[ "${1:-}" == "--probe" ]]; then
  python - <<'EOF'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 640, "height": 360})
    page.set_content("<h1 style='font-family:sans-serif'>proxy render check</h1><button>Search</button>")
    box = page.locator("button").bounding_box()
    page.screenshot(path="/tmp/render-check.png")
    browser.close()
print("chromium ok", box)
EOF
fi
