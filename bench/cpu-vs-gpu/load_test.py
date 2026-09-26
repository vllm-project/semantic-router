#!/usr/bin/env python3
"""Concurrent load driver: sustained QPS + latency under concurrency.

The ext_proc classifier path handles one request at a time (there is no batch
knob like an LLM server), so the real-world analog of "batch size / throughput"
is concurrency: N clients hitting the router at once.

Only HTTP 2xx responses contribute to QPS and the latency percentiles. A 5xx
from Envoy usually means ext_proc never classified the request, so counting it
as throughput would report a healthy QPS for a broken router; the failures are
reported separately instead and the caller decides.

Usage: load_test.py <url> <duration_s> <concurrency> <payload_file>
Prints: "<concurrency> <ok> <http_err> <conn_err> <qps> <p50> <p95> <p99>"
        (latency in ms, QPS over successful responses)
Exits 1 when no request succeeded.
"""

import sys
import threading
import time
import urllib.error
import urllib.request

url, duration, concurrency, payload_file = (
    sys.argv[1],
    float(sys.argv[2]),
    int(sys.argv[3]),
    sys.argv[4],
)
with open(payload_file, "rb") as f:
    payload = f.read()

stop = time.monotonic() + duration

# One result slot per worker, filled in place and merged after the join: no
# shared mutation, so no lock and no globals.
results = [([], 0, 0) for _ in range(concurrency)]


def worker(slot):
    latencies = []
    http_errors = 0
    conn_errors = 0
    while time.monotonic() < stop:
        t0 = time.monotonic()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=300).read()
            latencies.append((time.monotonic() - t0) * 1000.0)
        except urllib.error.HTTPError:
            http_errors += 1
        except Exception:
            conn_errors += 1
    results[slot] = (latencies, http_errors, conn_errors)


t_start = time.monotonic()
threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(concurrency)]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.monotonic() - t_start

lat = [value for latencies, _, _ in results for value in latencies]
http_err = sum(errors for _, errors, _ in results)
conn_err = sum(errors for _, _, errors in results)

n = len(lat)
if n == 0:
    print(f"{concurrency} 0 {http_err} {conn_err} 0 0 0 0")
    print(
        f"ERROR: no successful responses ({http_err} HTTP errors, "
        f"{conn_err} transport errors)",
        file=sys.stderr,
    )
    sys.exit(1)
lat.sort()
qps = n / elapsed


def p(q):
    return lat[min(n - 1, int(n * q))]


print(
    f"{concurrency} {n} {http_err} {conn_err} {qps:.1f} "
    f"{p(.5):.0f} {p(.95):.0f} {p(.99):.0f}"
)
