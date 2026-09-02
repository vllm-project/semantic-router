# TD051: Evaluation Worker Isolation Gap

## Status

Open

## Owner Plan

[PL-0039: Evaluation Plane](../plans/pl-0039-evaluation-plane.md)

## Release Relevance

The current worker is suitable for trusted first-party executors and
server-owned broker operations. Running third-party adapter code or treating
worker-authored artifacts as an independent security boundary remains blocked
until the worker has a distinct service identity and privilege boundary.

## Scope

- worker identity, groups, filesystem, network, and resource isolation;
- separation from Dashboard management authority and secrets;
- typed server-side artifact parsing and bounded worker output.

## Summary

The Dashboard launches Python in isolated mode with an allowlisted environment
and broker pipe. The worker applies resource limits, seccomp, Landlock,
`no_new_privs`, non-dumpable state, private filesystem roots, bounded output,
and process-group cancellation. Network, process creation, ptrace,
process-vm, pidfd, and signal escape paths are denied fail closed.

Those controls materially harden execution, but the worker still shares the
Dashboard container's real UID and service boundary. It therefore is not a
least-privilege home for untrusted benchmark code, especially when the parent
deployment has management credentials or a container-runtime group.

## Evidence

- The Go process launches `sandbox_worker.py` as an isolated subprocess and
  passes live authority only through a narrow broker pipe.
- The Python sandbox installs Landlock, seccomp, rlimits, `no_new_privs`, and
  non-dumpable state before executor work begins.
- The canonical Dashboard image still has one runtime non-root identity; the
  worker has no distinct UID, mount namespace, or independent service boundary.
- Public artifacts are allowlisted and parsed as bounded typed data by the Go
  control plane; worker-authored HTML or Markdown is not a public artifact.

## Why It Matters

Evaluation inputs can include private prompts, model outputs, tools, media, and
adversarial artifacts. Syscall and path restrictions reduce exposure but do not
replace a distinct identity, mount boundary, network policy, and authenticated
control plane.

## Desired End State

Evaluation runs in a dedicated worker identity or service with sealed inputs,
run-private output, explicit network and resource policy, and a narrow
authenticated control channel. Dashboard validates typed output and owns every
public rendering boundary.

## Exit Criteria

- Run evaluation workers under a distinct UID and group set with no access to
  Dashboard secrets, control sockets, configuration stores, or sibling runs.
- Give workers sealed read-only inputs, one run-private writable mount, explicit
  network policy, resource quotas, and a narrow authenticated broker channel.
- Add adversarial integration coverage for cross-run reads, secrets and socket
  access, oversized output, descendants, cancellation, and broker misuse.
- Demonstrate the boundary in the canonical container and restart workflow.
