#!/usr/bin/env python3
"""Concurrent load driver: sustained QPS + latency under concurrency.

The ext_proc classifier path handles one request at a time (there is no batch
knob like an LLM server), so the real-world analog of "batch size / throughput"
is concurrency: N clients hitting the router at once. This reports achieved QPS
and latency percentiles for a fixed prompt at a fixed concurrency.

Non-2xx HTTP responses (e.g. 503 "no decision matched") still count as processed
requests — the router ran classification either way, which is what throughput
measures here. Only connection errors / timeouts are dropped.

Usage: load_test.py <url> <duration_s> <concurrency> <payload_file>
Prints: "<concurrency> <completed> <qps> <p50> <p95> <p99>"  (latency in ms)
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

lat = []
lock = threading.Lock()
stop = time.monotonic() + duration


def worker():
    local = []
    while time.monotonic() < stop:
        t0 = time.monotonic()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=300).read()
            local.append((time.monotonic() - t0) * 1000.0)
        except urllib.error.HTTPError:
            local.append((time.monotonic() - t0) * 1000.0)
        except Exception:
            pass
    with lock:
        lat.extend(local)


t_start = time.monotonic()
threads = [threading.Thread(target=worker) for _ in range(concurrency)]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.monotonic() - t_start

n = len(lat)
if n == 0:
    print(f"{concurrency} 0 0 0 0 0")
    sys.exit()
lat.sort()
qps = n / elapsed


def p(q):
    return lat[min(n - 1, int(n * q))]


print(f"{concurrency} {n} {qps:.1f} {p(.5):.0f} {p(.95):.0f} {p(.99):.0f}")
