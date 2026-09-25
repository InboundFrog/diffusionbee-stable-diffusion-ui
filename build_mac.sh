#!/usr/bin/env bash
# One-command macOS arm64 build: backend venv (uv, Python 3.12) -> PyInstaller --onedir bundle
# -> electron-builder DMG in electron_app/dist_electron/.
#
# Unsigned by default. Signed when CSC_NAME (keychain identity, e.g. "Jane Doe (TEAMID)") or
# CSC_LINK/CSC_KEY_PASSWORD (.p12) is set; notarized when APPLE_ID + APPLE_APP_SPECIFIC_PASSWORD +
# APPLE_TEAM_ID, or APPLE_KEYCHAIN_PROFILE, or APPLE_API_KEY + APPLE_API_KEY_ID + APPLE_API_ISSUER are set.
#
#   ./build_mac.sh                 full build
#   ./build_mac.sh --backend-only  stop after bundling + smoke-testing the backend
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
OUT=$ROOT/dist                               # gitignored
BACKEND=$OUT/diffusionbee_backend            # -> BACKEND_BUILD_PATH -> Resources/core/
REQS=backends/stable_diffusion/requirements.txt

[ "$(uname -sm)" = "Darwin arm64" ] || { echo "error: build on an Apple Silicon Mac" >&2; exit 1; }
command -v uv >/dev/null || { echo "error: install uv (brew install uv)" >&2; exit 1; }

echo "==> backend venv ($OUT/venv)"
# Fresh venv every time, separate from the dev venv, so the bundle only has what REQS says.
uv venv --clear --python 3.12 "$OUT/venv"
uv pip install --python "$OUT/venv/bin/python" -r "$REQS" pyinstaller==6.22.3

echo "==> bundling backend ($BACKEND)"
"$OUT/venv/bin/pyinstaller" --noconfirm --log-level WARN \
  --distpath "$OUT" --workpath "$OUT/pyinstaller" packaging/diffusionbee_backend.spec

echo "==> smoke test"
"$OUT/venv/bin/python" packaging/smoke_test.py "$BACKEND/diffusionbee_backend"
du -sh "$BACKEND"

[ "${1:-}" = "--backend-only" ] && exit 0

echo "==> electron app"
# No identity configured -> don't let electron-builder pick a random dev cert from the keychain.
[ -n "${CSC_NAME:-}${CSC_LINK:-}" ] || export CSC_IDENTITY_AUTO_DISCOVERY=false
cd electron_app
npm ci
BACKEND_BUILD_PATH="$BACKEND" BUILD_ARCH=arm64 npm run electron:build
ls -lh dist_electron/*.dmg
