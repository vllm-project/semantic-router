#!/bin/sh
# One-time interactive login inside the isolated evaluation container.
# The token comes from stdin and is never written to this repository.
set -eu
stty -echo 2>/dev/null || true
IFS= read -r hf_task_token
[ -n "$hf_task_token" ] || exit 2
hf auth login --token "$hf_task_token" --no-add-to-git-credential --quiet
unset hf_task_token
