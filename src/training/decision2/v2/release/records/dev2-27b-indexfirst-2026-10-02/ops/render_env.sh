#!/usr/bin/env bash
# The card render environment of the earlier card rounds, rebuilt on a node (COORDINATION 2026-10-02 12:40: card
# rendering runs on the servers): CPython 3.12.13 (uv 0.12.0's standalone build), exactly the packages of the round-2
# environment (matplotlib 3.11.2, Pillow 12.3.0, numpy 2.5.3 and their pinned dependencies, installed without
# resolution), the six Inter 4.1 fonts of the official release zip and the vLLM-SR logo (website/static/img/artworks/
# vllm-sr-logo.dark.png at the pinned repository commit). Every download is checked by SHA-256; a finished
# environment is only verified. Prints the versions and digests.
# Usage (node A): bash <mirror>/v2/release/records/dev2-27b-indexfirst-2026-10-02/ops/render_env.sh [ROOT]
set -euo pipefail
T=${1:-/data/dev2/tools/card-render}
UV_URL=https://github.com/astral-sh/uv/releases/download/0.12.0/uv-x86_64-unknown-linux-gnu.tar.gz
INTER_URL=https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
INTER_SHA=9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e
LOGO_URL=https://raw.githubusercontent.com/vllm-project/semantic-router/efe400d2198e9909f41cea7ead77fa26d4a63f61/website/static/img/artworks/vllm-sr-logo.dark.png
LOGO_SHA=e6b8428eb67f9318450e4b938639055d20e5f3643aa3b6f676a7b96a7beda94a
PINS=(contourpy==1.4.0 cycler==0.12.1 fonttools==4.66.1 kiwisolver==1.5.1 matplotlib==3.11.2 numpy==2.5.3
  packaging==26.3 pillow==12.3.0 pyparsing==3.3.3 python-dateutil==2.9.0.post0 six==1.17.0)
FONTS="288316099b1e0a47a4716d159098005eef7c0066921f34e3200393dbdb01947f  Inter-Bold.ttf
97ad806f526e41546d46365bb3a393145f75b7b1568913db74549ad8b8dba872  Inter-Medium.ttf
40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82  Inter-Regular.ttf
78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3  Inter-SemiBold.ttf
b74c8e0dd744b3347faca4c96bc7b2e32f7d6f62300a79b1d1a99331e44a5bc4  InterDisplay-Bold.ttf
0310d7a325896129730c6c8cf9a6e0f81ee258bedf77b1ff059b2a7b75f74e02  InterDisplay-SemiBold.ttf"
fetch() {  # URL SHA256 DEST
  if [[ ! -f "$3" ]]; then
    curl -fsSL --retry 3 -o "$3.part" "$1"
    mv "$3.part" "$3"
  fi
  [[ "$(sha256sum < "$3" | cut -c1-64)" == "$2" ]] || { echo "$3 is not $2" >&2; rm -f "$3"; exit 1; }
}
mkdir -p "$T/dl" "$T/bin" "$T/fonts"
if [[ ! -x "$T/bin/uv" ]]; then
  curl -fsSL --retry 3 "$UV_URL" | tar -xz -C "$T/dl"
  install -m 755 "$T/dl/uv-x86_64-unknown-linux-gnu/uv" "$T/bin/uv"
fi
[[ "$("$T/bin/uv" --version)" == "uv 0.12.0"* ]] || { echo "uv is not 0.12.0" >&2; exit 1; }
export UV_PYTHON_INSTALL_DIR=$T/python UV_CACHE_DIR=$T/cache UV_NO_CONFIG=1
if [[ ! -x "$T/venv/bin/python" ]]; then
  "$T/bin/uv" python install 3.12.13
  "$T/bin/uv" venv --python 3.12.13 "$T/venv"
  "$T/bin/uv" pip install --python "$T/venv/bin/python" --no-deps "${PINS[@]}"
fi
want=$(printf '%s\n' "${PINS[@]}" | LC_ALL=C sort)
got=$("$T/bin/uv" pip freeze --python "$T/venv/bin/python" | LC_ALL=C sort)
[[ "$got" == "$want" ]] || { echo "the render venv holds other packages:"$'\n'"$got" >&2; exit 1; }
fetch "$INTER_URL" "$INTER_SHA" "$T/dl/Inter-4.1.zip"
while read -r sha font; do
  [[ -f "$T/fonts/$font" ]] || (cd "$T/fonts" && python3 -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); open(sys.argv[2], "wb").write(z.read("extras/ttf/" + sys.argv[2]))' "$T/dl/Inter-4.1.zip" "$font")
  [[ "$(sha256sum < "$T/fonts/$font" | cut -c1-64)" == "$sha" ]] || { echo "$font is not $sha" >&2; exit 1; }
done <<< "$FONTS"
fetch "$LOGO_URL" "$LOGO_SHA" "$T/logo.png"
"$T/venv/bin/python" -c 'import sys, matplotlib, PIL, numpy; print("render env", sys.version.split()[0], "matplotlib", matplotlib.__version__, "Pillow", PIL.__version__, "numpy", numpy.__version__)'
echo "fonts $(cd "$T/fonts" && sha256sum ./*.ttf | sha256sum | cut -c1-16) logo ${LOGO_SHA:0:16} root $T"
