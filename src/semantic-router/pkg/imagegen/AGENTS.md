# Image Generation Package Notes

## Scope

- `src/semantic-router/pkg/imagegen/**`

## Responsibilities

- Keep backend-specific request shaping, provider wiring, and compatibility support on separate seams.
- Treat `backend_vllm_omni.go` as a backend adapter hotspot, not as the package-default home for every image-generation behavior.
- Keep shared image-generation contracts distinct from backend-specific transport or payload translation.

## Change Rules

- `backend_vllm_omni.go` is a ratcheted hotspot. New provider-specific translation, fallback, or compatibility logic should move into adjacent helpers instead of widening that file.
- Do not mix backend configuration parsing with runtime request normalization when a dedicated adapter or helper can own one side.
- If a change touches both shared image-generation contracts and one backend adapter, keep the contract change narrow and backend-specific behavior local to that adapter seam.
