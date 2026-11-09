# Router and Engine modes

Use Router mode to route Chat, Responses and Messages requests to backend LLMs.
Use Engine mode to ask decision models native System One questions without
configuring Chat backends. Both modes use the same instance, listeners and model
deployments. On the docker target, Engine mode starts only the Router container;
`--dashboard` adds the Dashboard (and with it the host controller that
`vllm-sr instance` talks to), `--observability` adds Jaeger, Prometheus and
Grafana, and `--no-minimal` starts the whole stack. Without `--config`, and with
no `config.yaml` in the working directory, Engine mode keeps each model's
configuration and state in `~/.local/state/vllm-sr/engine/<model>`.

## Start an instance

```bash
# Serve a decision model through the native System One API.
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu

# Inspect the stack and its public native models.
vllm-sr status
curl -s localhost:8899/v1/systemone/models
```

With the Dashboard (`--dashboard`, or Router mode), `vllm-sr instance --config
<config> status` and `models` also report the instance's mode and model
readiness from its host controller.

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
That includes the model's revision, which stays when you upgrade `vllm-sr`:
serving the same model warns when the saved revision is not the one this
`vllm-sr` pins, and `--revision` switches it.

```bash
# Scale the selected model to two independent workers on two AMD GPUs.
vllm-sr serve --engine --platform rocm -dp 2 --device-ids 0,1

# Two workers sharing one AMD GPU; each needs memory for its own model.
vllm-sr serve --engine --platform rocm -dp 2 --device-ids 0
```

## Ask about images and videos

Decision 3.0 models (`vllm-sr/d3`, `d3-turbo`, `d3-ultra`, `d3-flash`,
`d3-mini`, `d3-nano`, `d3-lite`, `d3-edge`) read images and videos as well as text.
Serve one on an AMD Instinct MI325X:

```bash
vllm-sr serve vllm-sr/d3-lite --engine --platform rocm
```

A request may carry any number of `images` as base64 PNG, JPEG or WebP data
URLs and `videos` as base64 MP4, WebM, QuickTime or Matroska data URLs; every
question sees them, images first, then videos, in front of the text. Each
image may have up to 8,000,000 bytes and 16,000,000 pixels, and the model
reads it at up to 1.6 megapixels. Send the body on standard input, as below:
a command-line argument holds at most 128 KiB.

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "vllm-sr/d3-lite",
  "state": "What does the attached image show?",
  "images": ["data:image/png;base64,$(base64 -w0 chart.png)"],
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of image is this?",
             "criteria": {"chart": "A chart or plot", "photo": "A photograph", "document": "A document"}}
  }
}
EOF
```

Each video may have up to 32,000,000 bytes, 300 seconds and 8,294,400 pixels
per frame. The model reads 2 frames per second, at least 4 and at most 32
spread over the whole video, each at up to 200,704 pixels; the videos of one
request take at most 16,384 input tokens, and a request body may have up to
48 MiB:

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "vllm-sr/d3-lite",
  "state": "What happens in the clip?",
  "videos": ["data:video/mp4;base64,$(base64 -w0 clip.mp4)"],
  "questions": {
    "moving": {"type": "noul", "instructions": "Does something move across the scene?"}
  }
}
EOF
```

A malformed or over-limit image or video, or one sent to a model that does
not read it, is answered 400 `invalid_request` with the reason as the
message, such as `video longer than 300 s`, `image exceeds 8 MB` or
`model does not read videos`; a body over 48 MiB is answered 413
`request_too_large`. The runtime's model card lists the `modalities` a model
reads and its image and video `limits`.

## Find things in images and videos

`vllm-sr/d3-turbo` reads a request's state (its text, images and videos) once
and answers every question from that reading. Besides Choice, Noul and Score
it answers `box` questions, a box around what the instructions describe, and
`track` questions, a box for each moment of the request's only video (`rate`
1 or 2 per second). A box question with `at` reads the video at that second.
Serve it on an AMD Instinct MI325X:

```bash
vllm-sr serve vllm-sr/d3-turbo --engine --platform rocm
```

`vllm-sr/d3-ultra` answers the same questions the same way; a text-only
request also lists its questions in front of its state.

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "vllm-sr/d3-turbo",
  "state": "The customer attached the receipt and a clip of the counter.",
  "images": ["data:image/png;base64,$(base64 -w0 receipt.png)"],
  "videos": ["data:video/mp4;base64,$(base64 -w0 counter.mp4)"],
  "questions": {
    "total": {"type": "box", "instructions": "Where is the total amount on the receipt?"},
    "cup": {"type": "box", "instructions": "The paper cup", "at": 2.0},
    "hand": {"type": "track", "instructions": "The cashier's hand", "rate": 1}
  }
}
EOF
```

A box answer is `{"type": "box", "box": {"top", "bottom", "left", "right"},
"confidence"}` with each edge a fraction of the image's height or width, and
`time` added on a video; a track answer is `{"type": "track", "track": [{"time",
"box", "confidence", "visible"}, ...]}`. A video box or track question reads a
clip up to the model card's `box_video_max_seconds`, when the package sets it.

A question or an option may also hold an image that only that question sees:
name it in the text and send the image under `assets`, as base64 or as a part
of a `multipart/form-data` body:

```json
{
  "model": "vllm-sr/d3-turbo",
  "state": "A customer sent two pictures.",
  "assets": {
    "<picture 1>": {"type": "image", "data_base64": "iVBORw0KGgo..."},
    "<picture 2>": {"type": "image", "data_base64": "iVBORw0KGgo..."}
  },
  "questions": {
    "receipt": {"type": "choice", "instructions": "Which picture is a shop receipt?",
                "criteria": {"first": "<picture 1>", "second": "<picture 2>"}}
  }
}
```

Each question reads up to 32,768 tokens; a longer one is answered 400
`invalid_request` with reason `question_tokens`. `"options": {"long_context":
true}` raises that to the card's `max_input_tokens` (262,144), slower and
less accurate than the validated length, and nothing is truncated. The Router
waits 30 seconds for an answer, and for a long-context request up to the
listener's `systemone.max_deadline_ms` (10 minutes by default); `"deadline_ms"`
in the same `options` sets another wait within that limit. One MI325X answers
about 127,000 tokens in 32 seconds, 213,000 in 142 and 262,000 in 203, with
154 GiB of its memory in use at that length.

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
