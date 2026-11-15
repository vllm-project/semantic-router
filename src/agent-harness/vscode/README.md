# vLLM Semantic Router in VS Code

This extension puts the models a vLLM Semantic Router gateway publishes, such as `vllm-sr/auto`,
into VS Code's chat model picker. Pick one and chat, or run agent mode, on the gateway: each request
goes to the gateway's Chat Completions API, and the gateway chooses the model that answers. The status
bar then shows which model that was.

The extension does no routing of its own. It does not choose between models, it does not use or
change GitHub Copilot's own models, and it does not change Copilot policy. What `vllm-sr/auto` does is
set by the gateway's recipe.

## What you need

- VS Code 1.104 or later. The extension uses VS Code's language model chat provider API, which became
  stable in that release.
- A Router gateway with at least one entrypoint. The local Quickstart serves one at
  `http://localhost:8899`; see [Connect an agent harness](https://vllm-sr.ai/docs/installation/agent-harness).
- Node.js 20 or later and npm, to build the extension.

## Install from source

```bash
cd src/agent-harness/vscode
npm ci
npm run compile
npx @vscode/vsce package --skip-license --out vllm-sr.vsix
code --install-extension vllm-sr.vsix
```

Instead of `code --install-extension`, you can open the Extensions view, choose **...** › **Install
from VSIX...** and pick `vllm-sr.vsix`.

## Use it

1. Check that the gateway answers: `curl -sS http://localhost:8899/v1/models`.
2. If the gateway's listener has `api_keys`, run **vLLM Semantic Router: Set Gateway API Key** from
   the Command Palette and enter one of them.
3. Open the chat view, open the model picker and choose a model under **vLLM Semantic Router**. If it is
   not listed, run **Chat: Manage Language Models**: that editor lists every model VS Code knows of,
   and whether each one shows in the picker.

After each reply the status bar shows the model the gateway chose, for example
`qwen36-35b-a3b-off`. Hover over it for the details:

```text
vLLM Semantic Router, last reply
Entrypoint: vllm-sr/auto
Model: qwen36-35b-a3b-off
Decision: think_fast
Recipe: default
```

These come from the reply's [routing headers](https://vllm-sr.ai/docs/api/headers):
`x-vsr-selected-model`, `x-vsr-selected-decision`, `x-vsr-selected-recipe`, and
`x-vsr-response-path` when the answer came from somewhere other than the upstream model, such as
`cache`. `Decision: none` means the gateway named no decision, as for a model asked for by name. Click
the item to open the extension's log, which has one line per reply and the errors.

Other commands: **vLLM Semantic Router: Clear Gateway API Key**, **Refresh Models** (after the gateway's
entrypoints change) and **Show Log**.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| `vllm-sr.gatewayUrl` | `http://localhost:8899` | The gateway's inference address, without `/v1`. It is read from user or remote settings only, never from a workspace, so a workspace cannot send your API key elsewhere. |
| `vllm-sr.models` | `[]` | Model IDs from the gateway's `GET /v1/models` to show, in this order. Empty shows every entrypoint the gateway lists, which are the models whose `routing.resolution` is `virtual`. List a backend model's ID to show it as well. |
| `vllm-sr.maxInputTokens` | `128000` | The prompt size VS Code fits each request into. |
| `vllm-sr.maxOutputTokens` | `16384` | The reply size VS Code is told to expect. It is not sent to the gateway. |

The gateway does not publish context windows or output limits, so set the two token settings to the
smallest limits among the models an entrypoint can reach.

The API key is kept in VS Code's secret storage, never in `settings.json`, and the extension does not
write it to its log. It is sent as `Authorization: Bearer <key>`. An error the gateway answers is shown
and logged with the key and any bearer token replaced by `***`, in case the gateway echoes them.

## Limits

- Requests use Chat Completions (`POST /v1/chat/completions`), streamed. The Responses and Messages APIs
  are not used.
- Text and tool calls are sent and received. Images are not sent (the models do not offer image input),
  and tool results keep only their text and JSON. A model's reasoning text is not shown, since VS Code's
  stable API has no part for it.
- Token counts that VS Code asks for are estimates, about four characters a token.
- Each VS Code window sends one `x-session-id`, so the gateway can keep a conversation on its model.
  VS Code's stable API does not give the extension a conversation ID, so no `x-conversation-id` is sent,
  and the gateway tells conversations apart by how they start.
- The status bar shows the last reply in the window, from any chat.
- In a remote window (SSH, a container, WSL), `localhost` means the machine the extension runs on.
- In VS Code 1.104, models from extensions were offered only on individual GitHub Copilot plans. Check
  VS Code's [language model documentation](https://code.visualstudio.com/docs/agent-customization/language-models)
  for the release you run. Copilot Business and Enterprise administrators can turn models from
  extensions off with the Bring Your Own Language Model Key policy; this extension does not change it.
