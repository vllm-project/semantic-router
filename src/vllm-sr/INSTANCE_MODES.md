# Router and Engine modes

Use Router mode to route Chat, Responses and Messages requests to backend LLMs.
Use Engine mode to ask decision models native System One questions without
configuring Chat backends. Both modes use the same instance, Dashboard,
listeners and model deployments.

## Start an instance

```bash
# Serve a decision model through the native System One API.
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu

# Inspect the instance and its model readiness.
vllm-sr instance --config config.yaml status
vllm-sr instance --config config.yaml models
```

A new Engine instance publishes the selected model on port 8899. Discover its
public model ID with `GET /v1/systemone/models`, then send questions to
`POST /v1/systemone` or its alias `POST /v1/decisions`. Follow the
[quickstart](../../website/docs/model-runtime/quickstart.md) for a complete
request and response.

`--engine` (`-e`) selects Engine mode for that start. Every `serve` command
without it selects Router mode, including when the previous start used Engine
mode. To enable routing, configure your backend models and routing policy, then
start with that file:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --replace-active-config
```

Use `--replace-active-config` when replacing the saved local configuration with
a file you edited. Omit it on routine restarts to preserve Dashboard changes.
Starting with `--engine` again retains the saved routing policy for later use.

## Choose and scale a model

The optional `MODEL` argument replaces the artifact in the default deployment
selected by `global.model_catalog.system.decision_model.deployment`. Omitting
it keeps that deployment; a new configuration defaults to Vela 2.0 0.3B.
Only explicitly supplied model or placement options override saved settings.

```bash
# Scale the selected model to two independent workers on two AMD GPUs.
vllm-sr serve --engine --platform rocm -dp 2 --device-ids 0,1

# Two workers sharing one AMD GPU; each needs memory for its own model.
vllm-sr serve --engine --platform rocm -dp 2 --device-ids 0
```

## Ask about images and videos

Decision 3.0 models (`vllm-sr/d3`, `d3-flash`, `d3-mini`, `d3-nano`,
`d3-lite`, `d3-edge`) read images and videos as well as text. Serve one on an
AMD Instinct MI325X:

```bash
vllm-sr serve vllm-sr/d3-lite --engine --platform rocm
```

A request may carry any number of `images` as base64 PNG, JPEG or WebP data
URLs and `videos` as base64 MP4, WebM, QuickTime or Matroska data URLs; every
question sees them, images first, then videos, in front of the text. Each
image may have up to 8,000,000 bytes and 16,000,000 pixels, and the model
reads it at up to 1.6 megapixels:

```bash
IMAGE="data:image/png;base64,$(base64 -w0 chart.png)"
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3-lite",
  "state": "What does the attached image show?",
  "images": ["'"$IMAGE"'"],
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of image is this?",
             "criteria": {"chart": "A chart or plot", "photo": "A photograph", "document": "A document"}}
  }
}'
```

Each video may have up to 32,000,000 bytes, 300 seconds and 8,294,400 pixels
per frame. The model reads 2 frames per second, at least 4 and at most 32
spread over the whole video, each at up to 200,704 pixels; the videos of one
request take at most 16,384 input tokens, and a request body may have up to
48 MiB:

```bash
VIDEO="data:video/mp4;base64,$(base64 -w0 clip.mp4)"
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3-lite",
  "state": "What happens in the clip?",
  "videos": ["'"$VIDEO"'"],
  "questions": {
    "moving": {"type": "noul", "instructions": "Does something move across the scene?"}
  }
}'
```

A model that does not read a request's images or videos answers it with
`invalid_request`. The runtime's model card lists the `modalities` a model
reads and its image and video `limits`.

Use canonical YAML for additional deployments, task bindings, attached workers
and Kubernetes placement. See the [deployment guide](../../website/docs/model-runtime/deploy.md).
Model placement and replica count apply in either mode.

## Publish native models

An existing configuration must explicitly list public native model IDs under
`listeners[].systemone.models`. Those IDs come from each deployment's
`public_name`, or its Hub artifact ID when no public name is set. A local model
path needs an explicit `public_name`. Listener API keys apply to native requests.
The Chat `listeners[].models` allowlist is a separate setting.

Starting another mode or choosing another model preserves existing grants.
Update the allowlist deliberately when publishing a new model. Check
`/v1/systemone/models` for publication and `vllm-sr instance models` for actual
readiness. Worker APIs such as classify, embeddings, rerank and bundle belong
to directly operated `vllm-srun` workers; see the
[runtime reference](../../website/docs/model-runtime/reference.md).

## Troubleshooting

- **Chat requests fail in Engine mode:** start without `--engine`, using a
  configuration with backend models and routing.
- **A model is missing from native discovery:** check the listener's
  `systemone.models` list and the deployment's public name.
- **A published model cannot answer:** inspect `instance models` and
  `vllm-sr logs router` for loading or readiness errors.
- **Restarting does not pick up an edited file:** validate it, then use
  `--replace-active-config` to apply that file over the saved local state.

The Dashboard displays the startup mode and manages model deployments. Its
`/api/instance` and `/api/instance/models` endpoints provide inspection;
mode changes use `serve` at startup. The local host controller maintains
recovery state under `$XDG_STATE_HOME/vllm-sr/instances` (default
`~/.local/state/vllm-sr/instances`). Keep that state across controller restarts.
Kubernetes and external workers remain managed by their deployment system.
