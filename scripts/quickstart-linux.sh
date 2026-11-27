#!/usr/bin/env bash
# RWKV-Pi One-Command Local Deployment & Launcher Setup for Linux / WSL2
# Usage:
#   bash quickstart-linux.sh [options]
#
# Options:
#   --dir <DIR>            Install root directory (default: $HOME/rwkv-agent)
#   --precision <MODE>     w8a16 (default, ~7.2GB for 12-16GB GPUs) | bf16 (~13.4GiB for 24GB+ GPUs)
#   --cuda <VER>           12.9 (default) | 13.2
#   --deepseek-key <KEY>   Optional DeepSeek API key (enables web_search/web_fetch & /rwkv-subagent)
#   --hf-endpoint <URL>    Hugging Face base URL (default: https://hf-mirror.com)
#   --official-hf          Use https://huggingface.co instead of hf-mirror.com
#   --port <PORT>          Local inference HTTP port (default: 8000)
#   --api-key <KEY>        Local inference API password (default: rwkv-local)
#   --install-only         Download and configure everything without starting the GPU engine now
#   -h, --help             Show help

set -euo pipefail

INSTALL_DIR="${RWKV_AGENT_DIR:-$HOME/rwkv-agent}"
PRECISION="${RWKV_PRECISION:-w8a16}"
CUDA_VER="${RWKV_CUDA_VER:-12.9}"
DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PORT="${RWKV_PORT:-8000}"
API_KEY="${RWKV_API_KEY:-rwkv-local}"
PI_REPO="${RWKV_PI_REPO:-https://github.com/w1c2j3/rwkv-pi.git}"
ENGINE_VERSION="v1.8.15"
INSTALL_ONLY=0

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --cuda) CUDA_VER="$2"; shift 2 ;;
    --deepseek-key) DEEPSEEK_KEY="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="${2%/}"; shift 2 ;;
    --official-hf) HF_ENDPOINT="https://huggingface.co"; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "$PRECISION" in
  w8a16|bf16) ;;
  *) echo "[ERROR] --precision must be 'w8a16' or 'bf16' (got: $PRECISION)" >&2; exit 1 ;;
esac

case "$CUDA_VER" in
  12.9|13.2) ;;
  *) echo "[ERROR] --cuda must be '12.9' or '13.2' (got: $CUDA_VER)" >&2; exit 1 ;;
esac

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing required command '$1'. $2" >&2
    exit 1
  fi
}

echo "==> [1/6] Checking Linux / WSL2 prerequisites..."
need_cmd curl "Install via: sudo apt update && sudo apt install -y curl git tar"
need_cmd git "Install via: sudo apt update && sudo apt install -y git"
need_cmd tar "Install via: sudo apt update && sudo apt install -y tar"
need_cmd sha256sum "Install coreutils"
need_cmd node "Install Node.js >= 22.19 from https://nodejs.org or via nvm: nvm install 22"
need_cmd npm "npm comes bundled with Node.js"

if ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)'; then
  echo "[ERROR] Node.js >= 22.19.0 is required, found $(node -v). Upgrade via nvm install 22 or https://nodejs.org" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR/models" "$INSTALL_DIR/bin" "$INSTALL_DIR/logs" "$INSTALL_DIR/data"

download_file() {
  local url="$1"
  local dest="$2"
  if command -v aria2c >/dev/null 2>&1; then
    local dir base
    dir="$(dirname "$dest")"
    base="$(basename "$dest")"
    aria2c -x 16 -s 16 -c --auto-file-renaming=false --allow-overwrite=true -d "$dir" -o "$base" "$url"
  else
    curl -fL --retry 5 --retry-delay 2 -C - -H "User-Agent: curl/8.5" "$url" -o "$dest"
  fi
}

echo "==> [2/6] Preparing rwkv_lightning_cuda ($ENGINE_VERSION, CUDA $CUDA_VER)..."
ENGINE_TAR="rwkv-lightning-${ENGINE_VERSION}-linux-x64-cuda${CUDA_VER}.tar.gz"
ENGINE_URL="https://github.com/Alic-Li/rwkv_lightning_cuda/releases/download/${ENGINE_VERSION}/${ENGINE_TAR}"

if [ -x "$INSTALL_DIR/bundle/rwkv_lighting_cuda" ] && [ -f "$INSTALL_DIR/bundle/rwkv_vocab_v20230424.txt" ]; then
  echo "    Found existing engine bundle at $INSTALL_DIR/bundle, skipping download."
else
  curl -fsSL -H "User-Agent: curl/8.5" "${ENGINE_URL}.sha256" -o "$INSTALL_DIR/${ENGINE_TAR}.sha256"
  if [ ! -f "$INSTALL_DIR/$ENGINE_TAR" ] || ! (cd "$INSTALL_DIR" && sha256sum --check "${ENGINE_TAR}.sha256" >/dev/null 2>&1); then
    rm -f "$INSTALL_DIR/$ENGINE_TAR"
    download_file "$ENGINE_URL" "$INSTALL_DIR/$ENGINE_TAR"
  fi
  (cd "$INSTALL_DIR" && sha256sum --check "${ENGINE_TAR}.sha256")
  tar -xzf "$INSTALL_DIR/$ENGINE_TAR" -C "$INSTALL_DIR"
fi
LD_LIBRARY_PATH="$INSTALL_DIR/bundle/lib:${LD_LIBRARY_PATH:-}" "$INSTALL_DIR/bundle/rwkv_lighting_cuda" --version

echo "==> [3/6] Preparing RWKV-7 7.2B model weights ($PRECISION)..."
HF_REPO="RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource"
if [ "$PRECISION" = "w8a16" ]; then
  MODEL_FILE="RWKV7-Pi-7.2B-v0.1-final.w8a16.rwkvq"
  DECODE_BATCH=4
  CHUNK_SIZE=512
else
  MODEL_FILE="RWKV7-Pi-7.2B-v0.1-final.pth"
  DECODE_BATCH=16
  CHUNK_SIZE=1024
fi
MODEL_PATH="$INSTALL_DIR/models/$MODEL_FILE"

curl -fsSL -H "User-Agent: curl/8.5" "$HF_ENDPOINT/$HF_REPO/resolve/main/SHA256SUMS.txt" -o "$INSTALL_DIR/models/SHA256SUMS.txt"
EXPECTED_SHA="$(awk -v fn="$MODEL_FILE" '$2 == fn || $2 == ("*" fn) { print $1; exit }' "$INSTALL_DIR/models/SHA256SUMS.txt")"
if [ -z "$EXPECTED_SHA" ]; then
  echo "[ERROR] Could not find checksum for $MODEL_FILE in SHA256SUMS.txt" >&2
  exit 1
fi

verify_model() {
  [ -f "$MODEL_PATH" ] || return 1
  local actual
  actual="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
  [ "$actual" = "$EXPECTED_SHA" ]
}

if verify_model; then
  echo "    Verified existing model $MODEL_FILE ($EXPECTED_SHA)."
else
  echo "    Downloading $MODEL_FILE from $HF_ENDPOINT/$HF_REPO ..."
  download_file "$HF_ENDPOINT/$HF_REPO/resolve/main/$MODEL_FILE" "$MODEL_PATH"
  if [ "$PRECISION" = "w8a16" ]; then
    curl -fsSL -H "User-Agent: curl/8.5" \
      "$HF_ENDPOINT/$HF_REPO/resolve/main/${MODEL_FILE}.source.json" \
      -o "${MODEL_PATH}.source.json" || true
  fi
  echo "    Verifying SHA-256 checksum (may take 10-20s)..."
  if ! verify_model; then
    echo "[ERROR] Checksum mismatch for $MODEL_PATH! Delete the broken file and re-run." >&2
    exit 1
  fi
  echo "    Model checksum verified: $EXPECTED_SHA"
fi

echo "==> [4/6] Downloading built-in real-world demo benchmark suite (demos/)..."
DEMOS_TAR="$INSTALL_DIR/rwkv-pi-demos.tar.gz"
curl -fsSL -H "User-Agent: curl/8.5" "$HF_ENDPOINT/$HF_REPO/resolve/main/rwkv-pi-demos.tar.gz" -o "$DEMOS_TAR"
if [ ! -d "$INSTALL_DIR/demos/.templates" ]; then
  tar -xzf "$DEMOS_TAR" -C "$INSTALL_DIR"
  echo "    Extracted 7 real-world demo sandboxes to $INSTALL_DIR/demos/"
else
  echo "    Existing $INSTALL_DIR/demos/ preserved (run 'rwkv-demos reset' anytime to restore clean copies)."
fi

echo "==> [5/6] Installing and building rwkv-pi client..."
if [ ! -d "$INSTALL_DIR/rwkv-pi/.git" ]; then
  git clone "$PI_REPO" "$INSTALL_DIR/rwkv-pi"
fi
(cd "$INSTALL_DIR/rwkv-pi" && npm ci --no-audit --no-fund)

# Preserve existing DEEPSEEK_API_KEY in .env if re-running without --deepseek-key
if [ -z "$DEEPSEEK_KEY" ] && [ -f "$INSTALL_DIR/rwkv-pi/.env" ]; then
  DEEPSEEK_KEY="$(awk -F= '/^DEEPSEEK_API_KEY=/{sub(/^[^=]*=/,""); print; exit}' "$INSTALL_DIR/rwkv-pi/.env" || true)"
fi

DEFAULT_MODEL_ID="${MODEL_FILE%.rwkvq}"
DEFAULT_MODEL_ID="${DEFAULT_MODEL_ID%.pth}"

cat > "$INSTALL_DIR/engine.conf" <<EOF
INSTALL_DIR="$INSTALL_DIR"
MODEL_PATH="$MODEL_PATH"
VOCAB_PATH="$INSTALL_DIR/bundle/rwkv_vocab_v20230424.txt"
HOST="127.0.0.1"
PORT="$PORT"
API_KEY="$API_KEY"
DECODE_BATCH="$DECODE_BATCH"
CHUNK_SIZE="$CHUNK_SIZE"
EOF
chmod 600 "$INSTALL_DIR/engine.conf"

cat > "$INSTALL_DIR/rwkv-pi/.env" <<EOF
RWKV_BACKEND=lightning
RWKV_BASE_URL=http://127.0.0.1:${PORT}/v1
RWKV_MODEL=${DEFAULT_MODEL_ID}
RWKV_API_KEY=${API_KEY}
RWKV_TOOL_FORMAT=xml
RWKV_PROMPT_TEMPLATE=xml
RWKV_CONTEXT_WINDOW=32768
RWKV_MAX_TOKENS=8192
RWKV_HISTORY_MODE=epoch
RWKV_EPOCH_BUDGET_TOKENS=9600
RWKV_EPOCH_KEEP_RECENT_PAIRS=3
RWKV_EPOCH_MAX_RESULT_TOKENS=2000
RWKV_EDIT_MODE=text
RWKV_PLUGINS=none
RWKV_PI_DATA_ROOT=${INSTALL_DIR}/data

# ============================================================
# DeepSeek Official API Key (Optional)
# Used by BOTH native web search (web_search / web_fetch) AND /rwkv-subagent collaboration.
# Set via: rwkv-engine set-deepseek sk-xxxx   (or paste directly below)
# ============================================================
DEEPSEEK_API_KEY=${DEEPSEEK_KEY}
RWKV_WEB_API_KEY=${DEEPSEEK_KEY}
EOF
chmod 600 "$INSTALL_DIR/rwkv-pi/.env"

cat > "$INSTALL_DIR/bin/rwkv-engine" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SELF_DIR/../engine.conf"

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
PID_FILE="$INSTALL_DIR/logs/engine.pid"
LOG_FILE="$INSTALL_DIR/logs/engine.log"
ENV_FILE="$INSTALL_DIR/rwkv-pi/.env"

is_ready() {
  curl --noproxy "*" -fsS --max-time 2 -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/v1/models" >/dev/null 2>&1
}

served_model() {
  curl --noproxy "*" -fsS --max-time 3 -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/v1/models" \
    | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);process.stdout.write((j.data&&j.data[0]&&j.data[0].id)||"");}catch{}});'
}

cmd="${1:-status}"
arg2="${2:-}"

case "$cmd" in
  start)
    if is_ready; then
      [ "$arg2" = "--quiet" ] || echo "Engine already running at http://$HOST:$PORT/v1 (model: $(served_model))"
      exit 0
    fi
    [ "$arg2" = "--quiet" ] || echo "Starting rwkv_lighting_cuda in background (log: $LOG_FILE)..."
    mkdir -p "$INSTALL_DIR/logs"
    export LD_LIBRARY_PATH="$INSTALL_DIR/bundle/lib:${LD_LIBRARY_PATH:-}"
    nohup "$INSTALL_DIR/bundle/rwkv_lighting_cuda" \
      --model-path "$MODEL_PATH" \
      --vocab-path "$VOCAB_PATH" \
      --host "$HOST" \
      --port "$PORT" \
      --decode-batch-size "$DECODE_BATCH" \
      --chunk-size "$CHUNK_SIZE" \
      --wkv32 \
      --chunk-load \
      --cmix-sparse off \
      --state-db-path "$INSTALL_DIR/data/rwkv_sessions.db" \
      --password "$API_KEY" >> "$LOG_FILE" 2>&1 &
    eng_pid=$!
    echo "$eng_pid" > "$PID_FILE"
    for _ in $(seq 1 90); do
      if is_ready; then
        [ "$arg2" = "--quiet" ] || echo "Engine ready at http://$HOST:$PORT/v1 (PID $eng_pid, model: $(served_model))"
        exit 0
      fi
      if ! kill -0 "$eng_pid" 2>/dev/null; then
        echo "[ERROR] Engine exited unexpectedly. Last 30 log lines:" >&2
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
      fi
      sleep 1
    done
    echo "[ERROR] Timed out waiting for engine after 90s. Check $LOG_FILE" >&2
    exit 1
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      pid="$(cat "$PID_FILE")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 15); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 1
        done
        echo "Stopped rwkv_lighting_cuda (PID $pid); GPU VRAM released."
      else
        echo "Process $pid is not running."
      fi
      rm -f "$PID_FILE"
    else
      echo "No PID file found at $PID_FILE."
    fi
    ;;
  restart)
    "$0" stop || true
    "$0" start
    ;;
  status)
    if is_ready; then
      echo "ONLINE  http://$HOST:$PORT/v1  model=$(served_model)"
      curl --noproxy "*" -fsS -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/v1/server/status" && echo ""
    else
      echo "OFFLINE (run: rwkv-engine start)"
      exit 1
    fi
    ;;
  logs)
    tail -n 60 -f "$LOG_FILE"
    ;;
  set-deepseek)
    if [ -z "$arg2" ]; then
      echo "Usage: rwkv-engine set-deepseek <DEEPSEEK_API_KEY>" >&2
      exit 1
    fi
    node -e '
      const fs = require("node:fs");
      const [file, key] = process.argv.slice(1);
      let txt = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
      const upsert = (s, k, v) => {
        const re = new RegExp("^" + k + "=.*$", "m");
        return re.test(s) ? s.replace(re, k + "=" + v) : s.trimEnd() + "\n" + k + "=" + v + "\n";
      };
      txt = upsert(txt, "DEEPSEEK_API_KEY", key);
      txt = upsert(txt, "RWKV_WEB_API_KEY", key);
      fs.writeFileSync(file, txt, { mode: 0o600 });
    ' "$ENV_FILE" "$arg2"
    echo "Saved DEEPSEEK_API_KEY to $ENV_FILE (web_search, web_fetch, and /rwkv-subagent are now enabled)."
    ;;
  *)
    echo "Usage: rwkv-engine {start|stop|restart|status|logs|set-deepseek <KEY>}" >&2
    exit 1
    ;;
esac
EOF
chmod +x "$INSTALL_DIR/bin/rwkv-engine"

cat > "$INSTALL_DIR/bin/rwkv-demos" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SELF_DIR/../engine.conf"
DEMOS_DIR="$INSTALL_DIR/demos"
TPL_DIR="$DEMOS_DIR/.templates"

cmd="${1:-list}"
target="${2:-}"

case "$cmd" in
  list)
    echo "Built-in demo sandboxes in $DEMOS_DIR:"
    for d in "$TPL_DIR"/*; do
      [ -d "$d" ] && echo "  - $DEMOS_DIR/$(basename "$d")"
    done
    echo ""
    echo "Prompts guide: $DEMOS_DIR/PROMPTS.zh-CN.md"
    echo "Reset anytime: rwkv-demos reset"
    ;;
  reset)
    if [ ! -d "$TPL_DIR" ]; then
      echo "[ERROR] Template directory missing at $TPL_DIR" >&2
      exit 1
    fi
    for d in "$TPL_DIR"/*; do
      [ -d "$d" ] || continue
      name="$(basename "$d")"
      if [ -n "$target" ] && [ "$name" != "$target" ]; then
        continue
      fi
      node -e '
        const fs = require("node:fs");
        const [src, dst] = process.argv.slice(1);
        fs.rmSync(dst, { recursive: true, force: true });
        fs.cpSync(src, dst, { recursive: true });
      ' "$d" "$DEMOS_DIR/$name"
      echo "  [OK] Reset $DEMOS_DIR/$name"
    done
    ;;
  *)
    echo "Usage: rwkv-demos {list|reset [demo-name]}" >&2
    exit 1
    ;;
esac
EOF
chmod +x "$INSTALL_DIR/bin/rwkv-demos"

cat > "$INSTALL_DIR/bin/pi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SELF_DIR/../engine.conf"

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

# Load optional per-directory .rwkv-pi.env (e.g. demos/04-python-web-research)
if [ -f "./.rwkv-pi.env" ]; then
  while IFS='=' read -r k v; do
    k="${k#"${k%%[![:space:]]*}"}"
    [ -z "$k" ] || [[ "$k" == \#* ]] && continue
    if [ -z "${!k+x}" ]; then
      export "$k=$v"
    fi
  done < "./.rwkv-pi.env"
fi

ENV_FILE="$INSTALL_DIR/rwkv-pi/.env"
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then
  ds_from_env="$(awk -F= '/^DEEPSEEK_API_KEY=/{sub(/^[^=]*=/,""); print; exit}' "$ENV_FILE" || true)"
  if [ -n "$ds_from_env" ]; then
    export DEEPSEEK_API_KEY="$ds_from_env"
  fi
fi

if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  export RWKV_WEB_API_KEY="${RWKV_WEB_API_KEY:-$DEEPSEEK_API_KEY}"
  export RWKV_TOOLS="${RWKV_TOOLS:-read,write,edit,bash,grep,find,ls,web_search,web_fetch}"
else
  export RWKV_TOOLS="${RWKV_TOOLS:-read,write,edit,bash,grep,find,ls}"
fi

if ! command -v bwrap >/dev/null 2>&1; then
  export RWKV_SUBAGENT_SANDBOX="${RWKV_SUBAGENT_SANDBOX:-none}"
  export RWKV_SUBAGENT_MOUNT="${RWKV_SUBAGENT_MOUNT:-}"
fi

skip_engine=0
for arg in "$@"; do
  case "$arg" in
    -h|--help|-v|--version|login|logout|status|token) skip_engine=1 ;;
  esac
done

if [ "$skip_engine" -eq 0 ]; then
  "$SELF_DIR/rwkv-engine" start --quiet
  live_id="$(curl --noproxy "*" -fsS --max-time 3 -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/v1/models" \
    | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{try{const j=JSON.parse(s);process.stdout.write((j.data&&j.data[0]&&j.data[0].id)||"");}catch{}});')"
  if [ -n "$live_id" ]; then
    export RWKV_MODEL="$live_id"
  fi
fi

exec node --env-file="$ENV_FILE" "$INSTALL_DIR/rwkv-pi/dist/cli.js" "$@"
EOF
chmod +x "$INSTALL_DIR/bin/pi"

PATH_LINE="export PATH=\"$INSTALL_DIR/bin:\$PATH\""
for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$rc" ] && ! grep -Fq "$INSTALL_DIR/bin" "$rc"; then
    printf '\n# Added by rwkv-pi quickstart\n%s\n' "$PATH_LINE" >> "$rc"
    echo "    Registered $INSTALL_DIR/bin in $rc"
  fi
done

echo "==> [6/6] Finalizing setup..."
if [ "$INSTALL_ONLY" -eq 0 ]; then
  "$INSTALL_DIR/bin/rwkv-engine" start
else
  echo "    Skipped engine startup (--install-only)."
fi

cat <<EOF

============================================================
  RWKV-Pi Local Deployment Ready! (Linux / WSL2)
============================================================
  Install root : $INSTALL_DIR
  Demos suite  : $INSTALL_DIR/demos  (reset via: rwkv-demos reset)
  Config (.env): $INSTALL_DIR/rwkv-pi/.env
  Model        : $MODEL_FILE
  Engine API   : http://127.0.0.1:$PORT/v1

  1) Reload your shell PATH right now:
       export PATH="$INSTALL_DIR/bin:\$PATH"

  2) Optional: Configure DeepSeek API Key (for web_search & /rwkv-subagent):
       rwkv-engine set-deepseek sk-your-deepseek-key

  3) Run Pi in built-in real-world demos or ANY project folder:
       cd "$INSTALL_DIR/demos/01-portal-security" && pi

  4) Manage background GPU engine & demos:
       rwkv-engine status | logs | stop
       rwkv-demos list | reset
============================================================
EOF
