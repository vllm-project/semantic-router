# vLLM Semantic Router VS Code extension

- This folder is a VS Code extension (`vllm-project.vllm-sr-vscode`). It is a
  client of a Router gateway's public inference API (`GET /v1/models`,
  `POST /v1/chat/completions` and the `x-vsr-*` routing headers); it never
  imports router or runtime code, and nothing from a sibling folder.
- The gateway owns routing. The extension never chooses a model, never calls
  Copilot's models, and never touches Copilot settings or policy.
- Use only VS Code's stable API, at the `@types/vscode` version
  `package.json` pins, which is also the lowest `engines.vscode`. Raise both
  together.
- `src/gateway.ts` imports only vscode's types, so its tests run without VS
  Code; `src/extension.ts` is the only file that loads `vscode`.
- Keep the dependencies to TypeScript and type packages. Use `fetch`, not an
  HTTP library, and `node:test` for tests.
- Before a change is done, run `npm ci && npm test` here. CI runs it in
  `make agent-plugins-test`, through the `vscode-extension` entry in
  `tools/agent/domains.yaml`.
