# Developing vSR for OpenCode

Use Node 22.18 or later and the pinned OpenCode 2.2.0 SDK. From this directory:

```bash
npm ci
npm run check
npm test
npm run test:integration
```

`src/index.ts` owns host registration and cleanup. `settings.ts` validates options,
`runtime.ts` owns bounded HTTP calls and capability discovery, `decisions.ts`
retains session-scoped metadata, and `rpc.ts` defines server/UI messages.

`test:integration` requires the installed OpenCode 2.2.0 CLI and installs no
models. It packs/extracts the package and starts an isolated private server.

Tests use Node's built-in test runner, synthetic inputs and loopback HTTP
servers. Real runtime tests require an explicit separately started model. Do
not download weights, call paid APIs or consume user credentials in unit tests.
The lifecycle contract includes cancellation during response streaming and
queueing; neither should open the offline breaker. Test the installed packed
artifact through the real OpenCode CLI when changing exports or registration.

The package is self-contained and cannot import files beside its installed
root. The CLI resolves its plugin SDK imports in the host. A package changes
with its README and ownership/check registration. Development proceeds in
stages toward one integration Draft PR; intermediate foundations are not a
claim that the six-app product is complete.
