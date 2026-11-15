# Developing the VS Code extension

## Layout

| Path | What it holds |
| --- | --- |
| `package.json` | The manifest: the `languageModelChatProviders` entry (vendor `vllm-sr`), the commands, the settings, and the pinned TypeScript and type packages. |
| `src/extension.ts` | Activation: registers the chat provider, the status bar item, the commands and the log. The only file that loads `vscode`. |
| `src/gateway.ts` | The gateway client: the model list, the Chat Completions request built from VS Code's messages and tools, the streamed reply turned back into VS Code's parts, and the routing headers. |
| `src/test/gateway.test.ts` | `gateway.ts` against a fake gateway on a local port, with plain classes standing in for VS Code's parts. |

## The VS Code API

The extension uses the language model chat provider API: the `languageModelChatProviders`
contribution point, `vscode.lm.registerLanguageModelChatProvider`, and a `LanguageModelChatProvider`
with `provideLanguageModelChatInformation`, `provideLanguageModelChatResponse` and
`provideTokenCount`. VS Code 1.104 made it stable, and it is in `@types/vscode` from 1.104.0 on (not in
1.103.0). VS Code activates the extension when the chat needs the `vllm-sr` vendor's models, through
the activation event it derives from that contribution, so the manifest lists none.

A message can carry role 3, the System role of VS Code's proposed `languageModelSystem` API, which the
stable API does not name; `gateway.ts` maps it to `system` without using the proposed API.

## Run and check

```bash
npm ci
npm test
```

`npm test` compiles with `tsc` (type-checking `extension.ts` against the pinned VS Code API too) and runs
the tests with `node --test`. To try the extension without installing it, run `npm run compile` and
start VS Code with `code --extensionDevelopmentPath="$PWD"`, or package it as the README describes.

To see the requests the extension sends, point `vllm-sr.gatewayUrl` at the gateway and read the
extension's log (**vLLM Semantic Router: Show Log**), or compare with a request by hand:

```bash
curl -sS -i -N http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "vllm-sr/auto", "stream": true, "messages": [{"role": "user", "content": "Hello!"}]}'
```

## In this repository

CI's Agent Plugins contract runs `make agent-plugins-test` when this folder changes, as it does for the
Claude Code and Codex plugins; the target runs `npm ci` and `npm test` here. The `vscode-extension`
entry in `tools/agent/domains.yaml` maps the folder to that contract.
