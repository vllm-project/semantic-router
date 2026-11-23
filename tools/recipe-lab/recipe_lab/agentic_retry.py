"""Run a second, fresh coding session using only observable client failures."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .agentic_driver import (
    HeaderForwardProxy,
    atomic_write,
    completed_checkpoint,
    run_task,
)
from .contract import ContractError, dataset_content_hash


def retry_reason(result):
    """A completed client's observable outcome. No judge or benchmark labels."""
    if result.get("status") != "done":
        return None  # Repository preparation and other infrastructure errors.
    if result.get("exit_code") != 0 or (result.get("claude") or {}).get("is_error"):
        return "client_error"
    if not str(result.get("model_patch") or "").strip():
        return "empty_patch"
    return None


def _arm_identity_env(name):
    name = name.strip().upper()
    return name.startswith(
        ("ANTHROPIC_", "CLAUDE_CODE_OAUTH_", "CLAUDE_CODE_USE_")
    ) or name in {"CLAUDE_CODE_SUBAGENT_MODEL", "CLAUDE_CODE_API_KEY_HELPER_TTL_MS"}


def _check_address(value, *, clone=False):
    try:
        addr = urlsplit(value)
        if clone and "://" not in value:
            return  # Local path or git's SCP-like SSH syntax.
        if (not clone and addr.scheme not in {"http", "https"}) or (
            (not clone or addr.scheme in {"http", "https"}) and not addr.hostname
        ):
            raise ValueError
        if (
            (addr.username and addr.scheme != "ssh")
            or addr.password
            or addr.query
            or addr.fragment
        ):
            raise ValueError
        addr.port  # noqa: B018 - attribute access validates malformed ports
    except (TypeError, ValueError, AttributeError) as exc:
        raise ContractError(
            "retry run URLs must not contain embedded credentials or query strings"
        ) from exc


def _sum_known(values):
    if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values):
        return sum(values)
    return None


@contextmanager
def retry_executor(args):
    """Bind a policy and its task file to a resumable run before starting either arm."""
    # An interrupted process releases its OS lock. A competing writer must
    # never replace the first runner's policy, journals or task results.
    with (Path(args.out) / ".retry.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415 - Windows-only lock primitive

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415 - POSIX-only lock primitive

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ContractError(
                "retry output directory already has an active writer"
            ) from exc
        with _open_legs(args) as execute:
            yield execute


@contextmanager
def _open_legs(args):
    policy_file = Path(args.retry_policy)
    try:
        policy = json.loads(policy_file.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ContractError("retry policy must be valid UTF-8 JSON") from exc
    if not isinstance(policy, dict) or set(policy) - {
        "trigger",
        "target",
        "model",
        "api_key_env",
    }:
        raise ContractError(
            "retry policy expects trigger, target, optional model and api_key_env"
        )
    if policy.get("trigger") != "client_error_or_empty_patch":
        raise ContractError("unknown retry trigger")
    target = policy.get("target")
    _check_address(target)
    _check_address(args.target)
    _check_address(args.clone_url_template, clone=True)
    if "model" in policy and not isinstance(policy["model"], str):
        raise ContractError("retry policy model must be a string")
    env_key = policy.get("api_key_env")
    if env_key is not None and (
        not isinstance(env_key, str) or not os.environ.get(env_key)
    ):
        raise ContractError("retry policy api_key_env must name an existing credential")
    keys = {args.api_key, os.environ.get(env_key) if env_key else None} - {None, ""}
    for item in args.env or []:
        name, _, value = item.partition("=")
        if (
            _arm_identity_env(name)
            or (env_key and name.strip().upper() == env_key.upper())
            or value in keys
        ):
            raise ContractError(
                "use target, model and credential options to identify each retry arm"
            )
    client_env = {
        name: value
        for name, value in os.environ.items()
        if not _arm_identity_env(name)
        and value not in keys
        and (not env_key or name.upper() != env_key.upper())
    }
    out = Path(args.out)
    manifest = out / "retry-manifest.json"
    identity = {
        "schema": "recipe-lab-client-retry-v1",
        "policy_sha256": dataset_content_hash(policy_file),
        "tasks_sha256": dataset_content_hash(args.tasks),
        "primary_target": args.target,
        "primary_model": args.model,
        "primary_small_fast_model": args.small_fast_model,
        "max_turns": args.max_turns,
        "timeout_s": args.timeout_s,
        "max_output_tokens": args.max_output_tokens,
        "claude_bin": args.claude_bin,
        "clone_url_template_sha256": hashlib.sha256(
            args.clone_url_template.encode()
        ).hexdigest(),
        "env_sha256": hashlib.sha256(json.dumps(args.env or []).encode()).hexdigest(),
    }
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ContractError(
                "retry run inputs changed; choose a new output directory"
            )
    else:
        if (out / "results").exists() or (out / "attempts").exists():
            raise ContractError(
                "retry output already contains results without a matching policy"
            )
        atomic_write(out / "retry-policy.json", policy_file.read_bytes())
        atomic_write(manifest, json.dumps(identity, indent=2).encode())

    with ExitStack() as stack:
        legs = []
        for name in ("primary", "retry"):
            leg = copy.copy(args)
            leg.out = out / "attempts" / name
            leg.client_base_env = client_env
            if name == "retry":
                leg.target = target
                leg.model = leg.small_fast_model = policy.get("model")
                leg.api_key = os.environ[env_key] if env_key else "recipe-lab-dummy"
            proxy = HeaderForwardProxy(
                leg.target, leg.out / "requests", request_id_prefix=name + "-"
            )
            proxy.start()
            stack.callback(proxy.stop)
            legs.append((leg, proxy))

        def execute(task):
            inst = task["instance_id"]
            if (
                not isinstance(inst, str)
                or inst in {"", ".", ".."}
                or any(c in inst for c in "/\\:")
            ):
                raise ContractError("retry instance_id must be a single file name")
            _check_address(
                args.clone_url_template.format(repo=task["repo"]), clone=True
            )
            result_path = out / "results" / f"{inst}.json"
            cached = completed_checkpoint(result_path, inst)
            if cached is not None:
                return cached
            start_path = out / "attempts" / f"{inst}.start"
            if not start_path.exists():
                atomic_write(start_path, str(time.time()).encode())
            attempts = []
            journals = []
            reason = None
            for leg, proxy in legs:
                result = run_task(task, args=leg, proxy=proxy)
                attempts.append(result)
                journal = leg.out / "requests" / f"{inst}.jsonl"
                if journal.exists():
                    journals.append(journal.read_bytes())
                if len(attempts) == 1:
                    reason = retry_reason(result)
                    if reason is None:
                        break
            clients = [a.get("claude") or {} for a in attempts]
            usage = {
                k: _sum_known([(c.get("usage") or {}).get(k) for c in clients])
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            }
            final = {
                **attempts[-1],
                "seconds": round(time.time() - float(start_path.read_text()), 3),
                "session_ids": sorted(
                    {s for a in attempts for s in a.get("session_ids", [])}
                ),
                "claude": {
                    **clients[-1],
                    "usage": usage,
                    "total_cost_usd": _sum_known(
                        [c.get("total_cost_usd") for c in clients]
                    ),
                },
                "retry": {
                    "reason": reason,
                    "attempt_count": len(attempts),
                    "policy_sha256": identity["policy_sha256"],
                },
            }
            atomic_write(out / "requests" / f"{inst}.jsonl", b"".join(journals))
            atomic_write(result_path, json.dumps(final, ensure_ascii=False).encode())
            return final

        yield execute
