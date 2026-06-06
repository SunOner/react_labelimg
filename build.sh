#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
NODE_VERSION_FILE="$ROOT_DIR/.nvmrc"

CLEAN=0
FORCE_INSTALL=0
SKIP_INSTALL=0
RUN_LINT=0

log() {
  printf '\033[1;34m[build]\033[0m %s\n' "$*"
}

fail() {
  printf '\033[1;31m[build]\033[0m %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]

Options:
  --clean        Remove frontend/node_modules and frontend/dist before building.
  --install      Run npm ci even if node_modules already exists.
  --no-install   Skip dependency installation.
  --lint         Run npm run lint before npm run build.
  -h, --help     Show this help.

Run this from a Linux/WSL shell. In WSL, do not use Windows node/npm from /mnt/c.
EOF
}

while (($#)); do
  case "$1" in
    --clean)
      CLEAN=1
      FORCE_INSTALL=1
      ;;
    --install)
      FORCE_INSTALL=1
      ;;
    --no-install)
      SKIP_INSTALL=1
      ;;
    --lint)
      RUN_LINT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

is_wsl() {
  grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null
}

tool_path() {
  command -v "$1" 2>/dev/null || true
}

is_windows_tool() {
  local path
  path="$(tool_path "$1")"
  [[ "$path" == /mnt/[a-zA-Z]/* || "$path" == *.exe ]]
}

try_load_nvm() {
  local nvm_dir="${NVM_DIR:-$HOME/.nvm}"

  if [[ ! -s "$nvm_dir/nvm.sh" ]]; then
    return 0
  fi

  # shellcheck disable=SC1090
  . "$nvm_dir/nvm.sh"

  if [[ -f "$NODE_VERSION_FILE" ]]; then
    log "Using Node from .nvmrc ($(tr -d '[:space:]' < "$NODE_VERSION_FILE"))"
    nvm install --no-progress
    nvm use --silent
  fi
}

ensure_linux_node() {
  if is_wsl; then
    try_load_nvm
  fi

  if ! command -v node >/dev/null 2>&1; then
    fail "Node.js was not found. Install Linux Node in WSL, for example: nvm install && nvm use"
  fi

  if ! command -v npm >/dev/null 2>&1; then
    fail "npm was not found. Install Linux Node/npm in this shell."
  fi

  if is_wsl && { is_windows_tool node || is_windows_tool npm; }; then
    cat >&2 <<EOF
[build] Refusing to use Windows node/npm inside WSL.
[build] node: $(tool_path node)
[build] npm:  $(tool_path npm)
[build]
[build] Install/use Linux Node in WSL:
[build]   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
[build]   source ~/.nvm/nvm.sh
[build]   cd "$ROOT_DIR"
[build]   nvm install
[build]   ./build.sh --clean
EOF
    exit 1
  fi

  node - <<'EOF' || fail "Node $(node -v) is too old. Use Node 22 from .nvmrc or Node >=20.19."
const [major, minor] = process.versions.node.split('.').map(Number)
const ok = major > 22 || (major === 22 && minor >= 12) || (major === 20 && minor >= 19)
process.exit(ok ? 0 : 1)
EOF

  log "Node: $(node -v) ($(tool_path node))"
  log "npm:  $(npm -v) ($(tool_path npm))"
}

install_frontend_dependencies() {
  if ((SKIP_INSTALL)); then
    log "Skipping dependency install"
    return 0
  fi

  if ((CLEAN)); then
    log "Removing frontend/node_modules and frontend/dist"
    rm -rf -- "$FRONTEND_DIR/node_modules" "$FRONTEND_DIR/dist"
  fi

  if ((FORCE_INSTALL)) || [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    log "Installing frontend dependencies with npm ci"
    npm ci --prefix "$FRONTEND_DIR"
  else
    log "frontend/node_modules exists; skipping npm ci"
  fi
}

run_frontend_build() {
  if ((RUN_LINT)); then
    log "Running frontend lint"
    npm run lint --prefix "$FRONTEND_DIR"
  fi

  log "Building frontend"
  npm run build --prefix "$FRONTEND_DIR"
}

[[ -d "$FRONTEND_DIR" ]] || fail "frontend directory not found: $FRONTEND_DIR"

ensure_linux_node
install_frontend_dependencies
run_frontend_build

log "Done. Frontend build is in frontend/dist."
