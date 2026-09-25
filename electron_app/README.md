# Diffusion Bee Electron App

## Project setup
```
npm install
node node_modules/electron/install.js  # only if npm skipped electron's install script (npm 11 allowScripts)
```
Node 26 works as-is; no NODE_OPTIONS needed.

### Compiles and hot-reloads for development
```
 npm run electron:serve # run via electron 
```

### Compiles and minifies for production
```
npm run electron:build
```

Output: `dist_electron/DiffusionBee-<version>-arm64.dmg` (`BUILD_ARCH=x64` for Intel, `BACKEND_BUILD_PATH=<dir>` to bundle the python backend as `Resources/core`).
Unsigned local build: `CSC_IDENTITY_AUTO_DISCOVERY=false npm run electron:build`.
Signing/notarization is electron-builder's own: set `CSC_NAME` (or `CSC_LINK`) plus `APPLE_ID`/`APPLE_APP_SPECIFIC_PASSWORD`/`APPLE_TEAM_ID` (or `APPLE_API_KEY`/`APPLE_API_KEY_ID`/`APPLE_API_ISSUER`, or `APPLE_KEYCHAIN_PROFILE`).

In dev the backend is `PY_SCRIPT` (default `../backends/stable_diffusion/diffusionbee_backend.py`) run with `PYTHON` (default `python3`).

