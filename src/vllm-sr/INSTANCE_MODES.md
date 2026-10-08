# Managed instance modes

`vllm-sr serve --config config.yaml` attaches a host controller after a successful
local Docker or Podman startup with Dashboard enabled. It preserves Dashboard,
Router configuration, model caches and external inference services while an
operator changes the local data plane. Kubernetes and externally owned instances
remain owned by their deployment system and do not offer local mode switching.

```bash
vllm-sr instance --config config.yaml status
vllm-sr instance --config config.yaml models
vllm-sr instance --config config.yaml deploy --mode engine --deployment primary --request-id engine-1
vllm-sr instance --config config.yaml deploy --mode router --request-id router-1
```

Engine selects a declared `global.model_catalog.deployments` resource. It serves
native System One questions, without running Chat routing or a recipe. Router
mode runs the preserved Router configuration and can also publish System One.
Saving configuration while Engine runs does not start Router; explicitly deploy
Router mode to activate that configuration.

Both modes expose `POST /v1/systemone`, its native alias `POST /v1/decisions`, and
`GET /v1/systemone/models` through Dashboard's stable inference bridge. These
routes use the selected listener's `api_keys` and its separate, explicit
`systemone.models` grants. Chat `listener.models` grants do not grant System One
access. Public model names are deployment `public_name` values or Hub artifact
IDs; local artifacts require an explicit public name. With multiple listeners,
set `VLLM_SR_SYSTEMONE_LISTENER` to the intended listener when starting Dashboard.
Discovery describes publication, while `instance models` describes actual
readiness. A granted model that is not active in Engine returns unavailable;
deploying another model never automatically widens a grant.

Status reports desired mode separately from observed mode and the operation's
phase. It does not infer Router mode from a failed health probe. A repeated
request ID returns the recorded operation; a new ID can redeploy the same mode.
Readiness checks use actual model cards. A failed cutover restores retained
containers and the last working Router configuration. Interrupted operations
are recovered from the private journal before another operation is admitted.
No config hash or generation equality is a deployment gate.

The controller runs as the CLI user. Private state is stored under
`$XDG_STATE_HOME/vllm-sr/instances` (default `~/.local/state/vllm-sr/instances`).
Only its dedicated Unix socket directory is mounted read-only into Dashboard;
the Docker socket, controller manifest and journal are not mounted there.
Retain this host state and the same stack/config/state environment on restart.
For a host service supervisor, run the foreground command with the same user
and environment:

```bash
vllm-sr instance --config config.yaml controller
```

Normal `serve` and health repair preserve an intentional Engine mode. An
external health timer should observe nonterminal operations without starting
containers; in Engine mode it should inspect `instance models` for ready native
cards. The controller preserves prior containers for rollback; operators may
remove retained rollback containers after verifying a deployment.

An operator who uses the canonical lower-level container startup helper during
an image rollout must attach the controller after readiness, with the same stack
and state environment used for startup:

```bash
vllm-sr instance --config config.yaml attach --runtime-config /path/to/active/runtime-config.yaml --gateway standalone
```

`--runtime-config` must be the actual active config path and `--gateway` the
deployed Router gateway mode. This captures the running containers' immutable
image IDs; it does not start another data plane. After replacing the CLI package,
restart an idle controller through its host supervisor so future operations use
the new installed code. No browser request can supply a host command, image,
filesystem path or upstream URL to the controller.
