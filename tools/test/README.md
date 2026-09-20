# Test support

- `services/`: shared mock vLLM and MCP classifier servers used by E2E profiles
  and local development;
- `smoke/`: directly runnable API and local-stack checks, including the Milvus
  installation smoke;
- `soak/`: local stack orchestration for the soak runner in `e2e/cmd/soak`.

Scenario definitions and reusable E2E assertions remain in `e2e/`. Test services
are development fixtures, not supported production deployments.

`smoke/test-authz-rbac.sh` requires an already configured trusted authentication
stack, Authorino user secrets, and two real model backends. It checks authenticated
role routing and missing/invalid-token rejection. The maintained `authz-rbac`
E2E profile instead verifies that untrusted identity headers are stripped; it
does not provision the external identity system needed by this live smoke.
