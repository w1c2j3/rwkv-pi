#!/usr/bin/env bash
# Turnkey install: pinned dependencies, patched Pi source, pinned Lightning tokenizer, local config, readiness check.
# Safe to re-run; existing checkouts are verified against their locks and never reset.
#   scripts/install.sh            # then edit .env and run: npm run doctor && npm run start:source
set -euo pipefail
cd "$(dirname "$0")/.."
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)" >&2; exit 1; }; }
need node "Node.js >= 22.19"; need npm "comes with Node.js"; need git "source checkouts"; need g++ "builds the native tokenizer"
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)' \
  || { echo "Node.js >= 22.19 required, found $(node -v)" >&2; exit 1; }
npm ci
npm run pi:setup
npm run lightning:setup
npm run lightning:tokenizer
[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "created .env from .env.example; set RWKV_BASE_URL and RWKV_MODEL"; }
command -v bwrap >/dev/null || echo "note: install bubblewrap for harness runs and sandboxed subagents" >&2
npm run doctor -- --offline
echo "installed. Start the engine, then: npm run doctor && npm run start:source"
