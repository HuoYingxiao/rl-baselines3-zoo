#!/usr/bin/env bash

# Avoid WandB network/login blocking in batch runs
export WANDB_MODE="${WANDB_MODE:-offline}"

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Re-exec in the background with nohup so the run survives SSH disconnects.
if [[ -t 1 && "${RUN_SEQ_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/run_sequence_${ts}.log"
  echo "Starting detached run. Logs: ${log_file}"
  RUN_SEQ_CHILD=1 nohup "$0" "$@" >>"${log_file}" 2>&1 &
  disown
  exit 0
fi

list_file="a2c/anpg_scripts.txt"
targets=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -l|--list)
      list_file="$2"
      shift 2
      ;;
    *)
      targets+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${list_file}" ]]; then
  resolved_list="${list_file}"
  [[ -f "${resolved_list}" ]] || resolved_list="${SCRIPT_DIR}/${list_file}"

  if [[ ! -f "${resolved_list}" ]]; then
    echo "List file '${list_file}' not found." >&2
    exit 1
  fi

  list_dir="$(cd -- "$(dirname "${resolved_list}")" && pwd)"
  while IFS= read -r line; do
    [[ -z "${line}" || "${line}" =~ ^# ]] && continue
    # Resolve relative paths against the list file's directory.
    if [[ "${line}" = /* ]]; then
      targets+=("${line}")
    else
      targets+=("${list_dir}/${line}")
    fi
  done < "${resolved_list}"
fi

if [[ "${#targets[@]}" -lt 1 ]]; then
  echo "Usage: $(basename "$0") [-l list.txt] <script1.sh> [script2.sh ...]" >&2
  exit 1
fi

for target in "${targets[@]}"; do
  candidate="${target}"
  [[ -f "${candidate}" ]] || candidate="${SCRIPT_DIR}/${target}"

  if [[ ! -f "${candidate}" ]]; then
    echo "Skipping '${target}': file not found." >&2
    continue
  fi

  echo "[RunExp] $(date) Starting ${candidate}"
  bash "${candidate}"
  echo "[RunExp] $(date) Finished ${candidate}"
done

echo "[RunExp] $(date) All scripts complete."
