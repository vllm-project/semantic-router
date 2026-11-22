# vSR OpenCode plugin

- This package is self-contained. Consume the decision runtime over HTTP; do not
  import sibling harness plugins or router/runtime implementation modules.
- `src/index.ts` owns OpenCode registration and cleanup. Keep policy and HTTP
  code independent of the host so cancellation and failure paths can be tested.
- `src/rpc.ts` is the server/UI contract. Validate inputs at the server boundary.
  Return neither credentials nor prompt/tool-output text in diagnostics.
- Keep tests fixture-based. Live runtime/model tests require explicit input and
  must not download models as part of `npm test`.
- Run `make agent-opencode-test` from the repository root, or `npm ci`,
  `npm run check`, and `npm test` here. CI runs it in the Agent Plugins contract.
- When the Claude Code plugin changes how it reads a listener's profile or
  rules (`lib/profile.ts`, `lib/rules.ts`), port the change to `listener.ts`.
  When it bumps the runtime pin (`COMMIT` in `skills/setup/scripts/local.py`),
  bump the same commit in `skills/vsr-setup/SKILL.md`: both plugins install
  one shared runtime. When it changes the calibrate fit (`skills/calibrate/scripts`),
  port it to `calibration.ts` and regenerate `test/fixtures/calibration-golden.json`
  from its scripts. Test the packed artifact with
  the pinned OpenCode release whenever registration or package exports change.
