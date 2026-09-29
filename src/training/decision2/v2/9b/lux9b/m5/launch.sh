#!/usr/bin/env bash
# usage: launch.sh CHAIN FILE SIZE SHA256 [ARGS...]
# Node side, 16:00 chain rule: starts the bash script FILE (an uploaded chain file, or the
# mirror's chain-step.sh for a single step, with ARGS) only if its size in bytes and SHA-256
# equal SIZE and SHA256 (from upload_chain.sh). Runs it detached as `setsid nohup bash FILE
# ARGS...` with output to /data/dev2/runs/9b/m5/logs/CHAIN.console, writes the PID to
# logs/CHAIN.pid and a receipt to logs/CHAIN.launch.json, and prints the PID. Refuses a CHAIN
# whose PID is still alive or whose console log exists. Check it with alive.sh CHAIN.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
chain=$1; file=$2; size=$3; want=$4; shift 4
[[ "$chain" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad chain name $chain" >&2; exit 2; }
[ -s "$file" ] || { echo "$file is missing or empty" >&2; exit 3; }
got_size=$(stat -c %s "$file")
got_sha=$(sha256sum "$file" | cut -c1-64)
[ "$got_size" = "$size" ] && [ "$got_sha" = "$want" ] \
  || { echo "refused: $file is $got_size bytes sha256 $got_sha, expected $size $want" >&2; exit 3; }
logs=$M5/logs
mkdir -p "$logs"
pidf=$logs/$chain.pid
console=$logs/$chain.console
if [ -s "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
  echo "refused: chain $chain is running as PID $(cat "$pidf")" >&2; exit 4
fi
[ ! -e "$console" ] || { echo "refused: $console exists; pick a new chain name" >&2; exit 4; }
setsid nohup bash "$file" "$@" > "$console" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidf"
python3 - "$logs/$chain.launch.json" "$chain" "$file" "$got_size" "$got_sha" "$pid" "$@" <<'EOF'
import datetime, json, sys
out, chain, path, size, sha, pid, *args = sys.argv[1:]
json.dump({"chain": chain, "file": path, "size": int(size), "sha256": sha, "args": args, "pid": int(pid),
           "launched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
          open(out, "w"), indent=1)
EOF
sleep "${D2_LAUNCH_SETTLE:-2}"
if kill -0 "$pid" 2>/dev/null; then
  echo "chain $chain running: pid $pid session $(ps -o sid= -p "$pid" | tr -d ' ') console $console" >&2
else
  echo "chain $chain pid $pid already exited; see $console" >&2
fi
echo "$pid"
