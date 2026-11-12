# vSR OpenCode plugin

- This package is self-contained. Consume the decision runtime over HTTP; do not
  import sibling harness plugins or router/runtime implementation modules.
- `src/index.ts` owns OpenCode registration and cleanup. Keep policy and HTTP
  code independent of the host so cancellation and failure paths can be tested.
- `src/rpc.ts` is the server/UI contract. Validate inputs at the server boundary.
  Return neither credentials nor prompt/tool-output text in diagnostics.
- Keep tests fixture-based. Live runtime/model tests require explicit input and
  must not download models as part of `npm test`.
- Run `npm ci`, `npm run check`, and `npm test`. Test the packed artifact with
  the pinned OpenCode release whenever registration or package exports change.
