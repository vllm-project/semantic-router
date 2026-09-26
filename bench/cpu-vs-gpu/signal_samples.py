#!/usr/bin/env python3
"""Signal-extraction sample gate for the CUDA benchmarks.

Compares `llm_signal_extraction_latency_seconds_count` between two Prometheus
scrapes and requires every named signal to have recorded at least `min` new
samples. Without this gate a run whose ext_proc path never classified anything
still produces a plausible-looking latency/QPS report.

The floor is `min` rather than an exact match because one request emits at
least one sample per signal but may emit more: the jailbreak and PII paths
record a per-rule sample on top of the per-request `*_evaluated` one.

Usage: signal_samples.py <before.txt> <after.txt> <min> [signal,signal,...]
Prints "<signal> <delta>" per signal; exits 1 when any signal is short.
"""

import re
import sys

before_file, after_file, floor, *rest = sys.argv[1:]
minimum = int(floor)
signals = (rest[0] if rest else "domain,jailbreak,pii").split(",")


def counts(path):
    pattern = re.compile(
        r'llm_signal_extraction_latency_seconds_count\{.*signal_type="([^"]+)".*\}\s+([\d.eE+-]+)'
    )
    out = {}
    try:
        with open(path) as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    out[match.group(1)] = float(match.group(2))
    except OSError as err:
        print(f"ERROR: cannot read {path}: {err}", file=sys.stderr)
        sys.exit(1)
    return out


before, after = counts(before_file), counts(after_file)
short = []
for signal in signals:
    delta = after.get(signal, 0.0) - before.get(signal, 0.0)
    print(f"{signal} {delta:.0f}")
    if delta < minimum:
        short.append(f"{signal}={delta:.0f}")

if short:
    print(
        f"ERROR: signal samples below the expected floor of {minimum}: "
        + ", ".join(short),
        file=sys.stderr,
    )
    sys.exit(1)
