"""Check live same-host vLLM export, reuse, and recomputation with an explicit artifact."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import urllib.request
from pathlib import Path

from src.kv_connector.snapshot import LocalSnapshotStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifact / "manifest.json").read_text())
    compat = manifest["compatibility"]
    store = LocalSnapshotStore(args.snapshot_root)
    prefix = [100] * 128
    hint = {
        "namespace": "validation-tenant",
        "cache_id": "correctness-" + str(time.time_ns()),
        "mapper_id": manifest["mapper_id"],
    }

    def api(side, path, body=None):
        url = args.source_url if side == "source" else args.target_url
        req = urllib.request.Request(
            url.rstrip("/") + path,
            data=json.dumps(body).encode() if body else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return dict(resp.headers), resp.read().decode()

    def complete(side, prompt, transfer=None):
        body = {
            "model": compat[side + "_model"],
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": 1,
        }
        if transfer:
            body["kv_transfer_params"] = transfer
        started = time.perf_counter()
        headers, text = api(side, "/v1/completions", body)
        return {
            "seconds": time.perf_counter() - started,
            "headers": headers,
            "body": json.loads(text),
        }

    for side in ("source", "target"):
        for _ in range(180):
            try:
                api(side, "/health")
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError(side)
    source = complete("source", prefix, dict(hint, mapper_id=""))
    snapshot = store.load(
        hint["namespace"],
        hint["cache_id"],
        source_model=compat["source_model"],
        source_revision=compat["source_revision"],
    )
    assert len(snapshot.token_ids) == len(prefix)
    prompt = [*prefix, 101]
    cold = complete("target", prompt)
    assert cold["headers"].get("x-vsr-kv-transfer-status") is None
    cases = {
        "mapped": hint,
        "missing": dict(hint, cache_id="missing"),
        "wrong_mapper": dict(hint, mapper_id="unknown"),
        "cross_tenant": dict(hint, namespace="other-tenant"),
    }
    expired = dataclasses.replace(
        snapshot, cache_id=hint["cache_id"] + "-expired", expires_at=time.time() + 2
    )
    store.publish(expired)
    bad = dataclasses.replace(
        snapshot,
        cache_id=hint["cache_id"] + "-bad",
        layers={
            i: (k[:, :, :126].contiguous(), v[:, :, :126].contiguous())
            for i, (k, v) in snapshot.layers.items()
        },
    )
    store.publish(bad)
    cases["expired"] = dict(hint, cache_id=expired.cache_id)
    cases["malformed"] = dict(hint, cache_id=bad.cache_id)
    time.sleep(3)
    output = {
        "mapper_id": manifest["mapper_id"],
        "source": source,
        "cold": cold,
        "snapshot_tokens": len(snapshot.token_ids),
        "cases": {},
    }
    for name, transfer in cases.items():
        result = complete("target", prompt, transfer)
        output["cases"][name] = result
        print(
            name,
            result["headers"].get("x-vsr-kv-transfer-status"),
            result["headers"].get("x-vsr-kv-transfer-reason"),
            result["seconds"],
            flush=True,
        )
        args.output.write_text(json.dumps(output, indent=2))
        expected = "applied" if name == "mapped" else "fallback_reprefill"
        assert result["headers"].get("x-vsr-kv-transfer-status") == expected, (
            name,
            result,
        )
        if name != "mapped":
            assert (
                result["body"]["choices"][0]["text"]
                == cold["body"]["choices"][0]["text"]
            ), name
    print("LIVE_CORRECTNESS_PASSED", flush=True)


if __name__ == "__main__":
    main()
