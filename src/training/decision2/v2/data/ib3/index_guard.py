"""G0 / G0u for IB3 (prereg ``records/ib3-prereg-2026-10-01.md`` §3).

    python3 -m v2.data.ib3.index_guard reference|scan|controls ...   IB1's matcher, IB3's template strings
    python3 -m v2.data.ib3.index_guard url-reference --suite S [--suite ...] --out-dir REF
    python3 -m v2.data.ib3.index_guard url-scan --reference REF --candidates T --candidates D --out-dir OUT
    python3 -m v2.data.ib3.index_guard url-controls --suite S [--suite ...] --reference REF --out RECEIPT

G0u: every URL (``http(s)://…`` or ``www.…``) in any string leaf of every suite row, and every e-mail domain, is
normalized (lower case; scheme, leading ``www.``, trailing ``/`` and punctuation removed); a candidate group is dropped
when any URL in its data leaves equals a suite URL or its host equals a suite host. Sets hold 64-bit blake2b hashes;
the reference keeps the benchmark of each hash's first occurrence for the private receipt. Public receipts are counts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.ib1 import index_guard as guard
from v2.data.ib3.families import TEMPLATE_STRINGS
from v2.data.sources.common import sha

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'()\[\]{}|\\^`]+")
EMAIL_RE = re.compile(r"(?i)[\w.+-]+@([a-z0-9-]+(?:\.[a-z0-9-]+)+)")
TRAIL = ".,;:!?)]}'\"/"
CONTROLS = 500
CONTROL_SALT = "ib3-g0u-control-v1"
PERSON = {kind: f"ib3-g0u-{kind}".encode() for kind in ("url", "host")}


def h64(text: str, kind: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            text.encode("utf-8"), digest_size=8, person=PERSON[kind]
        ).digest(),
        "big",
    )


def norm_url(raw: str) -> tuple[str, str] | None:
    """(normalized URL, host) of one URL string, or None."""
    text = raw.strip().lower()
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)
    text = text.rstrip(TRAIL)
    if text.startswith("www."):
        text = text[4:]
    host = re.split(r"[/?#]", text, maxsplit=1)[0]
    host = host.rsplit("@", 1)[-1].split(":", 1)[0].rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if "." not in host or not text:
        return None
    return text, host


def urls_in(value: Any) -> Iterator[tuple[str, str]]:
    for leaf in guard.leaves(value):
        for match in URL_RE.findall(leaf):
            found = norm_url(match)
            if found:
                yield found


def hosts_in_emails(value: Any) -> Iterator[str]:
    for leaf in guard.leaves(value):
        for domain in EMAIL_RE.findall(leaf):
            host = domain.lower().rstrip(".")
            yield host[4:] if host.startswith("www.") else host


def suite_values(row: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    urls, hosts = set(), set()
    for part in (row.get("state"), row.get("questions")):
        for url, host in urls_in(part):
            urls.add(url)
            hosts.add(host)
        hosts |= set(hosts_in_emails(part))
    return urls, hosts


def candidate_urls(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    data, _ = guard.candidate_leaves(row)
    return sorted({found for leaf in data for found in urls_in(leaf)})


# --------------------------------------------------------------------------- reference / scan / controls


def url_reference(paths: Sequence[Path], out: Path) -> dict[str, Any]:
    names: dict[str, int] = {}
    seen: dict[str, dict[int, int]] = {"url": {}, "host": {}}
    rows = rows_with = 0
    for line in guard.read_suite(paths):
        row = json.loads(line)
        rows += 1
        code = names.setdefault(guard.suite_name(row), len(names))
        urls, hosts = suite_values(row)
        rows_with += bool(urls or hosts)
        for kind, values in (("url", urls), ("host", hosts)):
            for value in values:
                seen[kind].setdefault(h64(value, kind), code)
    out.mkdir(mode=0o700)
    meta = {
        "rows": rows,
        "rows_with_a_url_or_host": rows_with,
        "units": {kind: len(values) for kind, values in seen.items()},
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    payload = {
        **meta,
        "benchmarks": sorted(names, key=names.get),
        "url": {str(k): v for k, v in seen["url"].items()},
        "host": {str(k): v for k, v in seen["host"].items()},
    }
    fd = os.open(out / "reference.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
    return meta


def load(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "reference.json").read_text(encoding="utf-8"))
    for kind in ("url", "host"):
        payload[kind] = {int(k): v for k, v in payload[kind].items()}
    return payload


def flags(
    found: Sequence[tuple[str, str]], ref: Mapping[str, Any]
) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {"url": set(), "host": set()}
    for url, host in found:
        if (code := ref["url"].get(h64(url, "url"))) is not None:
            out["url"].add(code)
        if (code := ref["host"].get(h64(host, "host"))) is not None:
            out["host"].add(code)
    return {kind: sorted(codes) for kind, codes in out.items() if codes}


def url_scan(paths: Sequence[Path], reference: Path, out: Path) -> dict[str, Any]:
    ref = load(reference)
    names = ref["benchmarks"]
    population: collections.Counter = collections.Counter()
    by_family: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    groups: dict[str, set[str]] = collections.defaultdict(set)
    flagged = []
    for path in paths:
        for line in path.read_bytes().decode("utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            family = row["family"]
            population[(family, row["split"])] += 1
            found = candidate_urls(row)
            by_family[family]["rows_with_url"] += bool(found)
            hit = flags(found, ref)
            if hit:
                groups[family].add(row["group_id"])
                by_family[family]["rows"] += 1
                for kind in hit:
                    by_family[family][f"rows_{kind}"] += 1
                flagged.append(
                    {
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "family": family,
                        "benchmarks": sorted(
                            {names[c] for codes in hit.values() for c in codes}
                        ),
                        "kinds": sorted(hit),
                    }
                )
    drop = sorted({g for members in groups.values() for g in members})
    families = sorted({f for f, _ in population})
    out.mkdir(mode=0o700)
    public = {
        "schema": "decision2.ib3.index-url-guard.v1",
        "rule": "a group is dropped if any URL in its data leaves equals a suite URL or its host equals a suite host",
        "reference": {
            k: ref[k] for k in ("rows", "rows_with_a_url_or_host", "units", "inputs")
        },
        "candidates": {
            f: {s: population.get((f, s), 0) for s in ("train", "select")}
            for f in families
        },
        "flagged": {
            f: {**dict(sorted(by_family[f].items())), "groups_dropped": len(groups[f])}
            for f in families
        },
        "groups_dropped": len(drop),
        "candidate_inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    public["drop_groups_sha256"] = guard.write_new(
        out / "drop-groups.txt", "".join(g + "\n" for g in drop)
    )
    private = {
        "schema": "decision2.ib3.index-url-guard-private.v1",
        "flagged": flagged,
        "hits_by_benchmark": dict(
            sorted(
                collections.Counter(
                    b for item in flagged for b in item["benchmarks"]
                ).items()
            )
        ),
    }
    guard.write_new(
        out / "index-url-guard.public.json",
        json.dumps(public, indent=1, sort_keys=True) + "\n",
    )
    guard.write_new(
        out / "index-url-guard.private.json",
        json.dumps(private, indent=1, sort_keys=True) + "\n",
    )
    return {"groups_dropped": len(drop)}


def perturb(raw: str) -> str:
    text = raw.strip()
    scheme = re.match(r"(?i)^(https?)://", text)
    rest = text[scheme.end() :] if scheme else text
    if rest.lower().startswith("www."):
        rest = rest[4:]
    else:
        rest = "www." + rest
    host, sep, tail = rest.partition("/")
    new_scheme = (
        "https://" if not scheme or scheme.group(1).lower() == "http" else "http://"
    )
    return new_scheme + host.upper() + sep + tail + ("" if text.endswith("/") else "/")


def url_controls(paths: Sequence[Path], reference: Path) -> dict[str, Any]:
    ref = load(reference)
    picked: dict[str, str] = {}
    for line in guard.read_suite(paths):
        row = json.loads(line)
        for part in (row.get("state"), row.get("questions")):
            for leaf in guard.leaves(part):
                for match in URL_RE.findall(leaf):
                    found = norm_url(match)
                    if found and found[0] not in picked:
                        picked[found[0]] = match
    chosen = sorted(picked.items(), key=lambda kv: sha(f"{CONTROL_SALT}:{kv[0]}"))[
        :CONTROLS
    ]

    def flagged(raw: str) -> bool:
        row = {"state": {"url": raw}, "instructions": "", "options": []}
        return bool(flags(candidate_urls(row), ref))

    exact = sum(flagged(raw) for _, raw in chosen)
    perturbed = sum(flagged(perturb(raw)) for _, raw in chosen)
    return {
        "controls": len(chosen),
        "exact_copies_flagged": exact,
        "perturbed_copies_flagged": perturbed,
        "pass": len(chosen) == CONTROLS and exact == CONTROLS and perturbed == CONTROLS,
    }


def main(argv: list[str] | None = None) -> int:
    guard.TEMPLATE_STRINGS = TEMPLATE_STRINGS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    one = sub.add_parser("url-reference")
    one.add_argument("--suite", type=Path, action="append", required=True)
    one.add_argument("--out-dir", type=Path, required=True)
    two = sub.add_parser("url-scan")
    two.add_argument("--reference", type=Path, required=True)
    two.add_argument("--candidates", type=Path, action="append", required=True)
    two.add_argument("--out-dir", type=Path, required=True)
    three = sub.add_parser("url-controls")
    three.add_argument("--suite", type=Path, action="append", required=True)
    three.add_argument("--reference", type=Path, required=True)
    three.add_argument("--out", type=Path, required=True)
    command = (argv if argv is not None else sys.argv[1:])[:1]
    if command not in (["url-reference"], ["url-scan"], ["url-controls"]):
        return guard.main(argv)
    args = parser.parse_args(argv)
    if args.command == "url-reference":
        print(json.dumps(url_reference(args.suite, args.out_dir)))
        return 0
    if args.command == "url-scan":
        print(json.dumps(url_scan(args.candidates, args.reference, args.out_dir)))
        return 0
    result = url_controls(args.suite, args.reference)
    guard.write_new(args.out, json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
