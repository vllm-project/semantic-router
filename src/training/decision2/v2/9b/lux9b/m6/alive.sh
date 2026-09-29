#!/usr/bin/env bash
# usage: alive.sh CHAIN
# Node side: liveness of a chain started by launch.sh, by PID (kill -0 on logs/CHAIN.pid) and by
# container (docker ps for d2-9b-m6-* job containers and dev2-9b-m6-* eval-runner containers),
# never by a process-name pattern. Prints the first and last lines of logs/CHAIN.log (the
# chain-step log) and logs/CHAIN.console. Exit 0 if the PID is alive, 1 if not.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
chain=$1
logs=$M6/logs
pidf=$logs/$chain.pid
[ -s "$pidf" ] || { echo "no PID file $pidf" >&2; exit 2; }
pid=$(cat "$pidf")
if kill -0 "$pid" 2>/dev/null; then
  alive=0; echo "chain $chain: running pid $pid elapsed $(ps -o etime= -p "$pid" | tr -d ' ')"
else
  alive=1; echo "chain $chain: dead (pid $pid)"
fi
echo "containers:"
dry docker ps --filter name=d2-9b-m6- --filter name=dev2-9b-m6- --format '  {{.Names}} {{.Status}}'
for f in "$logs/$chain.log" "$logs/$chain.console"; do
  if [ -s "$f" ]; then
    echo "$f: $(wc -l < "$f") lines"
    echo "  first: $(head -n 1 "$f")"
    tail -n 3 "$f" | sed 's/^/  last: /'
  else
    echo "$f: missing or empty"
  fi
done
exit $alive
