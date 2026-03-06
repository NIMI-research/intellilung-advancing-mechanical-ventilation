#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${1:?Usage: $0 <ssh_host_or_ssh_config_host> <config_path_in_repo> [git_ref] }"
CONFIG_PATH="${2:?Usage: $0 <ssh_host_or_ssh_config_host> <config_path_in_repo> [git_ref] }"
GIT_REF="${3:-main}"

REMOTE_REPO="workspace/INTELLILUNG_2/AI"

ssh "$SSH_HOST" bash -lc "'
  set -euo pipefail
  cd \"$REMOTE_REPO\"
  git fetch --all --prune
  git checkout \"$GIT_REF\"
  git pull --ff-only
  python hyper_param_tune/hparam_submit.py submit \"$CONFIG_PATH\"
'"