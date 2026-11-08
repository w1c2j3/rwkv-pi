#!/usr/bin/env bash
# Source this file to select this project's pinned pi in the current terminal.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Use: source scripts/activate-pi.sh\n' >&2
  exit 2
fi
rwkv_pi_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return
pi() {
  local rwkv_pi_env_args=()
  if [[ -f "$rwkv_pi_project_root/.env" ]]; then
    rwkv_pi_env_args+=("--env-file=$rwkv_pi_project_root/.env")
  fi
  RWKV_PI_SOURCE="$rwkv_pi_project_root/.upstream/pi" \
    node "${rwkv_pi_env_args[@]}" "$rwkv_pi_project_root/scripts/run-pi-source.mjs" \
    "$rwkv_pi_project_root/dist/cli.js" "$@"
}
printf 'pi now uses rwkv-pi in this terminal. Run: pi --offline --thinking off\n'
