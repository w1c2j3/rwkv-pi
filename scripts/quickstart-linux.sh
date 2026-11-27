#!/usr/bin/env bash
# RWKV-Pi one-command local deployment for Linux / WSL2.
#
#   curl -fsSL https://hf-mirror.com/RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource/resolve/main/quickstart-linux.sh | bash
#   curl -fsSL .../quickstart-linux.sh | bash -s -- --precision bf16 --dir /data/rwkv-agent
#
# The whole script lives inside main() so bash parses it completely before
# running anything; this keeps `curl | bash` safe from children reading stdin.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash quickstart-linux.sh [options]   (or: curl -fsSL <url> | bash -s -- [options])

  --dir <DIR>            Install root (default: $HOME/rwkv-agent)
  --precision <MODE>     auto (default: bf16 if GPU >= 22 GB, else w8a16) | w8a16 | bf16
  --cuda <VER>           12.9 (default) | 13.2
  --deepseek-key <KEY>   Optional DeepSeek API key (web_search/web_fetch + /rwkv-subagent)
  --hf-endpoint <URL>    Hugging Face base URL (default: https://hf-mirror.com)
  --official-hf          Use huggingface.co and nodejs.org instead of China mirrors
  --port <PORT>          Local inference HTTP port (default: 8000)
  --api-key <KEY>        Local inference API password (default: rwkv-local)
  --install-only         Install everything without starting the GPU engine
  -h, --help             Show this help
USAGE
}

die() { echo "[ERROR] $*" >&2; exit 1; }

main() {
INSTALL_DIR="${RWKV_AGENT_DIR:-$HOME/rwkv-agent}"
PRECISION="${RWKV_PRECISION:-auto}"
CUDA_VER="${RWKV_CUDA_VER:-12.9}"
DEEPSEEK_KEY="${DEEPSEEK_API_KEY:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
NODE_MIRROR="${NODE_MIRROR:-https://npmmirror.com/mirrors/node}"
NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmmirror.com}"
PORT="${RWKV_PORT:-8000}"
API_KEY="${RWKV_API_KEY:-rwkv-local}"
PI_REPO="${RWKV_PI_REPO:-https://github.com/w1c2j3/rwkv-pi.git}"
ENGINE_VERSION="v1.8.15"
INSTALL_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --cuda) CUDA_VER="$2"; shift 2 ;;
    --deepseek-key) DEEPSEEK_KEY="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="${2%/}"; shift 2 ;;
    --official-hf) HF_ENDPOINT="https://huggingface.co"; NODE_MIRROR="https://nodejs.org/dist"; NPM_REGISTRY="https://registry.npmjs.org"; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "Unknown argument: $1" ;;
  esac
done

case "$PRECISION" in auto|w8a16|bf16) ;; *) die "--precision must be auto, w8a16 or bf16 (got: $PRECISION)" ;; esac
case "$CUDA_VER" in 12.9|13.2) ;; *) die "--cuda must be 12.9 or 13.2 (got: $CUDA_VER)" ;; esac
[ "$(uname -m)" = "x86_64" ] || die "Only x86_64 Linux is supported (got: $(uname -m))."

echo "==> [1/6] Checking prerequisites..."
local missing=() c
for c in curl git tar unzip sha256sum; do
  command -v "$c" >/dev/null 2>&1 || missing+=("$c")
done
if [ ${#missing[@]} -gt 0 ]; then
  command -v apt-get >/dev/null 2>&1 || die "Please install: ${missing[*]}"
  echo "    Installing ${missing[*]} via apt (sudo may ask for your password)..."
  sudo apt-get update -qq && sudo apt-get install -y "${missing[@]/sha256sum/coreutils}"
fi

mkdir -p "$INSTALL_DIR/models" "$INSTALL_DIR/bin" "$INSTALL_DIR/logs" "$INSTALL_DIR/data"

# Node.js >= 22.19: reuse the system one if new enough, otherwise install a private copy (no sudo).
node_ok() {
  command -v node >/dev/null 2>&1 &&
    node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=19)?0:1)' 2>/dev/null
}
[ -x "$INSTALL_DIR/node/bin/node" ] && export PATH="$INSTALL_DIR/node/bin:$PATH"
if node_ok; then
  echo "    Node.js $(node -v) OK"
else
  echo "    Node.js >= 22.19 not found; installing private Node 22 into $INSTALL_DIR/node ..."
  local sums node_tgz
  sums="$(curl -fsSL "$NODE_MIRROR/latest-v22.x/SHASUMS256.txt")"
  node_tgz="$(printf '%s\n' "$sums" | awk '$2 ~ /^node-v22\.[0-9.]+-linux-x64\.tar\.gz$/ { print $2; exit }')"
  [ -n "$node_tgz" ] || die "Could not resolve the latest Node 22 package from $NODE_MIRROR"
  curl -fL --retry 5 "$NODE_MIRROR/latest-v22.x/$node_tgz" -o "$INSTALL_DIR/$node_tgz"
  (cd "$INSTALL_DIR" && printf '%s\n' "$sums" | awk -v f="$node_tgz" '$2 == f' | sha256sum --check --status) ||
    die "Node.js checksum mismatch for $node_tgz"
  mkdir -p "$INSTALL_DIR/node"
  tar -xzf "$INSTALL_DIR/$node_tgz" -C "$INSTALL_DIR/node" --strip-components=1
  rm -f "$INSTALL_DIR/$node_tgz"
  export PATH="$INSTALL_DIR/node/bin:$PATH"
  node_ok || die "Private Node.js install failed."
  echo "    Node.js $(node -v) installed"
fi
command -v bwrap >/dev/null 2>&1 || echo "    (optional) bubblewrap not found: /rwkv-subagent falls back to workspace-copy isolation."

# Pick weights by FREE memory on GPU 0 (the engine's device): BF16 needs ~14.2 GB, W8A16 ~8.5 GB.
local gpu_info vram_total="" vram_free="" need_mib
gpu_info="$(nvidia-smi --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
if [ -n "$gpu_info" ]; then
  vram_total="$(echo "$gpu_info" | cut -d, -f1 | tr -d ' ')"
  vram_free="$(echo "$gpu_info" | cut -d, -f2 | tr -d ' ')"
  echo "    GPU 0 memory: ${vram_free} MiB free / ${vram_total} MiB total"
else
  echo "    [WARN] nvidia-smi not found; make sure an NVIDIA driver with CUDA $CUDA_VER support is installed."
fi
if [ "$PRECISION" = "auto" ]; then
  if [ -n "$vram_free" ] && [ "$vram_free" -ge 16000 ]; then PRECISION="bf16"; else PRECISION="w8a16"; fi
  echo "    Auto-selected precision: $PRECISION"
fi
need_mib=9000; [ "$PRECISION" = "bf16" ] && need_mib=16000
if [ -n "$vram_free" ] && [ "$vram_free" -lt "$need_mib" ]; then
  echo "    [WARN] $PRECISION needs ~$((need_mib / 1000)) GB free VRAM; close other GPU apps before running pi."
fi

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

HF_REPO="RWKV-Agent-Lab/RWKV7-Agent-7.2B-OpenSource"
HF_BASE="$HF_ENDPOINT/$HF_REPO/resolve/main"
SUMS="$INSTALL_DIR/models/SHA256SUMS.txt"
curl -fsSL -H "User-Agent: curl/8.5" "$HF_BASE/SHA256SUMS.txt" -o "$SUMS"
sha_of() {
  awk -v fn="$1" '$2 == fn || $2 == ("*" fn) { print $1; exit }' "$SUMS"
}
sha_ok() {
  [ -f "$1" ] && [ "$(sha256sum "$1" | awk '{print $1}')" = "$2" ]
}

echo "==> [2/6] Preparing rwkv_lightning_cuda ($ENGINE_VERSION, CUDA $CUDA_VER)..."
ENGINE_ZIP="rwkv-lightning-${ENGINE_VERSION}-linux-x64-cuda${CUDA_VER}.zip"
if [ -x "$INSTALL_DIR/bundle/rwkv_lighting_cuda" ] && [ -f "$INSTALL_DIR/bundle/rwkv_vocab_v20230424.txt" ]; then
  echo "    Found existing engine bundle at $INSTALL_DIR/bundle, skipping download."
else
  ENGINE_SHA="$(sha_of "engine/$ENGINE_ZIP")"
  [ -n "$ENGINE_SHA" ] || die "Checksum for engine/$ENGINE_ZIP not found in SHA256SUMS.txt"
  if ! sha_ok "$INSTALL_DIR/$ENGINE_ZIP" "$ENGINE_SHA"; then
    download_file "$HF_BASE/engine/$ENGINE_ZIP" "$INSTALL_DIR/$ENGINE_ZIP"
    sha_ok "$INSTALL_DIR/$ENGINE_ZIP" "$ENGINE_SHA" || die "Engine archive checksum mismatch; delete $INSTALL_DIR/$ENGINE_ZIP and re-run."
  fi
  unzip -qo "$INSTALL_DIR/$ENGINE_ZIP" -d "$INSTALL_DIR"
  rm -f "$INSTALL_DIR/$ENGINE_ZIP"
fi
LD_LIBRARY_PATH="$INSTALL_DIR/bundle/lib:${LD_LIBRARY_PATH:-}" "$INSTALL_DIR/bundle/rwkv_lighting_cuda" --version

echo "==> [3/6] Preparing RWKV-7 7.2B model weights ($PRECISION)..."
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

EXPECTED_SHA="$(sha_of "$MODEL_FILE")"
[ -n "$EXPECTED_SHA" ] || die "Could not find checksum for $MODEL_FILE in SHA256SUMS.txt"

verify_model() {
  sha_ok "$MODEL_PATH" "$EXPECTED_SHA"
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
  GIT_TERMINAL_PROMPT=0 git clone --depth 1 "$PI_REPO" "$INSTALL_DIR/rwkv-pi"
else
  GIT_TERMINAL_PROMPT=0 git -C "$INSTALL_DIR/rwkv-pi" pull --ff-only || echo "    [WARN] Could not update rwkv-pi; keeping the existing copy."
fi
(cd "$INSTALL_DIR/rwkv-pi" && npm ci --no-audit --no-fund --registry "$NPM_REGISTRY")

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
[ -x "$INSTALL_DIR/node/bin/node" ] && export PATH="$INSTALL_DIR/node/bin:$PATH"

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
    # Run from data/ so the engine's ./uploads scratch dir never lands in the user's project.
    cd "$INSTALL_DIR/data"
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
      --password "$API_KEY" < /dev/null >> "$LOG_FILE" 2>&1 &
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
[ -x "$INSTALL_DIR/node/bin/node" ] && export PATH="$INSTALL_DIR/node/bin:$PATH"
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
[ -x "$INSTALL_DIR/node/bin/node" ] && export PATH="$INSTALL_DIR/node/bin:$PATH"

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

local rc_file="$HOME/.bashrc"
case "${SHELL:-}" in */zsh) rc_file="$HOME/.zshrc" ;; esac
if ! grep -Fqs "$INSTALL_DIR/bin" "$rc_file"; then
  printf '\n# Added by rwkv-pi quickstart\nexport PATH="%s/bin:$PATH"\n' "$INSTALL_DIR" >> "$rc_file"
  echo "    Registered $INSTALL_DIR/bin in $rc_file"
fi

echo "==> [6/6] Finalizing setup..."
if [ "$INSTALL_ONLY" -eq 0 ]; then
  "$INSTALL_DIR/bin/rwkv-engine" start
else
  echo "    Skipped engine startup (--install-only)."
fi

cat <<EOF

============================================================
  RWKV-Pi is ready (Linux / WSL2)
============================================================
  Install root : $INSTALL_DIR
  Model        : $MODEL_FILE
  Config (.env): $INSTALL_DIR/rwkv-pi/.env
  Engine API   : http://127.0.0.1:$PORT/v1

  Next: open a new terminal (or run: source $rc_file), then
       cd "$INSTALL_DIR/demos/01-portal-security" && pi

  Optional DeepSeek key (web_search + /rwkv-subagent):
       rwkv-engine set-deepseek <DEEPSEEK_API_KEY>
  Manage: rwkv-engine status|logs|stop   rwkv-demos list|reset
============================================================
EOF
}

main "$@"
