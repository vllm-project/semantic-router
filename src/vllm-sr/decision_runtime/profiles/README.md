# Decision model profiles

`vela/` and `qwen35/` contain one data-only template per model. A template owns
the manifest location, prompt policy, calibration fallback, input limit, dtype,
and initial physical batch size. The catalog supplies the canonical model ID
and selected revision. The family artifact selector reads the selected
snapshot's manifest to choose model files, including a complete Qwen weight
layout; changing weight shards does not require editing a packaged file list.

Hardware execution capability is not a model-file property. It lives in
`../backend_capabilities.py` and the family loaders. This avoids a second,
potentially contradictory `qualified` flag in every model profile. A backend
is exposed only when an owned executor and target-specific validation exist;
detecting a device does not qualify it. Promotion of an image or a model/device
pair still needs the release qualification and performance evidence.

To add a model, register its exact identity and family in the canonical catalog,
add a template to that family's directory, and test the artifact selection,
prompt semantics, and supported backend. To add a family, add its input encoder,
owned loader, row executor and image environment before registering templates.
To add hardware, implement and qualify its executor in the backend capability
layer; do not add a claim to a model template. Per-model and per-target batch
defaults may change only after numerical, memory, and performance validation.
