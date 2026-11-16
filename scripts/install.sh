#!/usr/bin/env bash
# Turnkey install. Safe to re-run; existing checkouts are verified against their locks and never reset.
#   scripts/install.sh --user   users: dependencies + the rwkv-pi command in ~/.local/bin; then run `rwkv-pi`
#                               (the first launch asks for a connection token)
#   scripts/install.sh          developers: also the patched Pi source, the pinned Lightning tokenizer, .env and an
#                               offline doctor; then edit .env and run: npm run doctor && npm run start:source
set -euo pipefail
cd "$(dirname "$0")/.."
user=0; [ "${1:-}" = --user ] && user=1
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)" >&2; exit 1; }; }
need node "Node.js >= 22.19 from https://nodejs.org"; need npm "comes with Node.js"
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)' \
  || { echo "Node.js >= 22.19 required, found $(node -v)" >&2; exit 1; }
if [ $user = 1 ]; then
  npm ci --no-audit --no-fund
  bin="${XDG_BIN_HOME:-$HOME/.local/bin}"; mkdir -p "$bin"
  ln -sfn "$PWD/dist/cli.js" "$bin/rwkv-pi"; chmod +x dist/cli.js
  case ":$PATH:" in *":$bin:"*) ;; *) echo "note: add $bin to PATH, e.g. echo 'export PATH=\"$bin:\$PATH\"' >> ~/.bashrc" >&2;; esac
  echo "installed. Run: rwkv-pi"
  exit 0
fi
need git "source checkouts"; need g++ "builds the native tokenizer"
npm ci
npm run pi:setup
npm run lightning:setup
npm run lightning:tokenizer
[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "created .env from .env.example; set RWKV_BASE_URL and RWKV_MODEL"; }
command -v bwrap >/dev/null || echo "note: install bubblewrap for harness runs and sandboxed subagents" >&2
npm run doctor -- --offline
echo "installed. Start the engine, then: npm run doctor && npm run start:source"
