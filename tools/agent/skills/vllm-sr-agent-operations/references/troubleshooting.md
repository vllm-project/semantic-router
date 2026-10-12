# Troubleshooting

Find the symptom, apply its fix, and rerun the step of the
[main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md) that failed. `vllm-sr logs router` and
`vllm-sr logs dashboard` print the last 200 lines; add `-f` to follow.
`vllm-sr status` reports what runs and whether setup or a restart is pending.

## Install

| Symptom | Cause and fix |
| --- | --- |
| `The virtual environment was not created successfully because ensurepip is not available` | Ubuntu or Debian without `python3-venv`. With the user's OK, `sudo apt-get install -y python3-venv`, then rerun the installer. |
| `vllm-sr: command not found` after the install | The launcher's directory isn't on `PATH`: `export PATH="$HOME/.local/bin:$PATH"`. |
| `vllm-sr serve --help` has no `--gateway` | The CLI predates standalone mode (`0.4.0` does). Install `--channel dev` as the main skill's step 3 says. |
| `pip install --pre vllm-sr` installs `0.4.0`, not a dev build | Dev versions sort below the release they follow ([#4695](https://github.com/vllm-project/semantic-router/issues/4695)). Use the installer's `--channel dev`, which pins the newest dev build. |
| `permission denied while trying to connect to the Docker daemon socket` | The account isn't in the `docker` group. Ask the user to add it (`sudo usermod -aG docker "$USER"`, then a new login session). Don't run `vllm-sr` with `sudo`: its state files would belong to root. |

## Start

| Symptom | Cause and fix |
| --- | --- |
| `serve` prints `Waiting for setup` and doesn't return | No complete `config.yaml`, so the stack is in Dashboard setup mode. Stop waiting (Ctrl-C), run `vllm-sr stop`, and check the directory: setup mode writes its own `config.yaml` there. Replace that file with the main skill's configuration and serve with `--config config.yaml`. |
| `… port N is already in use` | Something else publishes that port (8090 is the stack's sr-bench service). `docker ps` and `ss -ltnp` show the owner. Ask the user before stopping anything, or start this stack with a `VLLM_SR_PORT_OFFSET` ([Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#stack-identity)). |
| `config validate` fails | It names the field. `vllm-sr config schema --section PATH` shows what that section accepts. |
| `config validate` says only the CLI's own checks ran | It found no Router image or container runtime, so the Router's own validation didn't run. `vllm-sr config validate --config config.yaml --endpoint http://127.0.0.1:8080` asks the running Router instead. |
| `serve` times out | Image pulls, model downloads or GPU kernel compilation took longer than the budget. Read `vllm-sr logs router`, then rerun with `--startup-timeout 3600`. The containers stay up for inspection after a timeout. |
| `Platform 'amd' selected but missing AMD GPU devices` | `/dev/kfd` or `/dev/dri` isn't visible, so the models fall back to the CPU. Check `ls -l /dev/kfd /dev/dri` and that the ROCm driver is loaded on the host. |
| NVIDIA: `could not select device driver "" with capabilities: [[gpu]]` | Docker has no NVIDIA runtime. The NVIDIA Container Toolkit is missing; ask the user to install it. |
| `--platform rocm/cuda needs a Linux host` | macOS runs the Docker target on the CPU only. Serve without `--platform`. |

## Requests

| Symptom | Cause and fix |
| --- | --- |
| Status `502`, `503` or `504`, or the request hangs | The Router can't reach the backend, or it is slow. Check the endpoint from a container, as the main skill's step 2 does: a server on the host must listen beyond 127.0.0.1 and be addressed as `host.docker.internal:PORT`. `vllm-sr logs router` shows the upstream error. |
| `route probe`: `Read timed out. (read timeout=120.0)` | The backend is still generating. Pass a cap such as `--max-completion-tokens 256` with a prompt that asks for a short answer, and `--timeout` for slow hosts. |
| `route probe` fails with `Completion was truncated at the token or context limit` (`finish_reason: length`) | The answer hit the cap, which the probe counts as failed delivery. Raise `--max-completion-tokens` or ask for a shorter answer. |
| `x-vsr-selected-model` names an unexpected model, or no `x-vsr-*` headers | The request named a concrete model rather than `vllm-sr/auto` (concrete names bypass routing), or it didn't go through the Router's listener. Use `GET /v1/models` to see the public names. |
| A request that a model signal should match takes the fallback route | The model is still loading, or its threshold isn't met. The traced preview tells them apart ([Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#gpu-platforms)): `"state": "unknown"` with a `signal_error` is loading; a real score below the predicate is the threshold. |
| `vllm-sr logs envoy`: `This stack runs in standalone mode, with no Envoy container` | Expected in standalone mode. Envoy runs only with `--gateway extproc`. |

## Changes to a running stack

| Symptom | Cause and fix |
| --- | --- |
| Editing `config.yaml` changes nothing | The Docker stack serves `.vllm-sr/runtime-config.yaml`. Apply edits with `vllm-sr config apply --config config.yaml` ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md)). |
| `config apply`: `Restart required: run vllm-sr serve to apply.` | The change touches a listener (with `--gateway extproc`, also the provider topology). `config apply` saved it; `vllm-sr serve --config config.yaml` applies it. |
| `config apply`: `Router management API request timed out after 120s` | The change can still activate. Check `vllm-sr config versions` before retrying. |
| `status`: `Restart required` | A change saved in the Dashboard or with `vllm-sr config apply` needs `vllm-sr serve --config config.yaml` to apply it. |

## Kubernetes

| Symptom | Cause and fix |
| --- | --- |
| `Helm chart directory not found` | A pip or curl install carries no chart, and every `--target kubernetes` command looks in `./deploy/helm/semantic-router`. Pull the chart there and run from that directory ([Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#kubernetes)). |
| `'helm' is required for Kubernetes deployment` | Install `helm` 3, and `kubectl`, with the user's OK. |
| `Kubernetes deployment does not support Dashboard setup-mode configs` | Write a complete configuration first. |
| Pods run, requests fail upstream | The backend address must resolve inside the cluster; `host.docker.internal` doesn't. |
