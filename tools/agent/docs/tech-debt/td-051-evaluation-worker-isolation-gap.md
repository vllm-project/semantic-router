# TD051: Evaluation Worker Isolation Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

Dashboard evaluation is limited to trusted built-in executors and server-owned
targets. Executing third-party adapter code or treating worker artifacts as a
security boundary is blocked until this gap is closed.

## Scope

- worker process identity, groups, filesystem, network, and resource isolation
- separation from Dashboard container-management privileges
- server-side artifact parsing, redaction, and rendering
- bounded logs and event payloads

## Summary

The Dashboard starts the Python evaluation worker as a subprocess. Environment
allowlisting, process groups, timeouts, output caps, private stores, and public
artifact filename allowlists reduce exposure, but the worker still shares the
Dashboard process UID and supplementary groups. In deployments where the
Dashboard can access a container runtime socket, a compromised worker inherits
that authority. A worker-created checksum also proves consistency, not trusted
content.

## Evidence

- The current container drops the Dashboard to one non-root account after
  adding any required container-runtime group.
- The worker has no distinct UID, mount namespace, network policy, seccomp
  profile, or independent service boundary.
- Markdown and HTML reports are private; the public UI consumes a strictly
  parsed typed report. Other downloadable structured artifacts still require
  server-side schema and redaction enforcement.

## Why It Matters

Evaluation workloads may contain private prompts, model outputs, tools, media,
or adversarial data. Process-level validation is not a substitute for a least-
privilege execution boundary, especially when the parent Dashboard has
deployment-management authority.

## Desired End State

Evaluation executes in a dedicated worker service or sandbox with a distinct
identity, no container socket, sealed read-only inputs, a private output mount,
explicit network policy, resource quotas, and a narrow authenticated control
channel. The Dashboard validates typed outputs and renders public reports itself.

## Exit Criteria

- Run workers under a distinct UID and group set with no access to Dashboard
  secrets, control sockets, configuration stores, or sibling run directories.
- Apply read-only source/input mounts, a run-private writable mount, no-new-
  privileges, syscall/resource limits, and per-suite network policy.
- Accept only versioned typed result/event schemas with count, line, total-byte,
  and redaction limits.
- Parse and validate every public structured artifact server-side; render human
  documents from typed data instead of trusting worker-authored markup.
- Add adversarial integration tests for cross-run reads, secret exfiltration,
  socket access, oversized output, descendant processes, and cancellation.
