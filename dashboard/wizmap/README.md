# Knowledge Map WizMap Shell

This directory vendors the `wizmap` frontend and adapts it for the vLLM Semantic Router dashboard.

## Purpose

- build a self-hosted Knowledge Map app that is served by the dashboard backend at `/embedded/wizmap/`
- load per-KB data from router-owned same-origin endpoints instead of the public demo datasets
- keep the WizMap runtime isolated from the main React dashboard bundle

## Build

```bash
cd dashboard/wizmap
npm install
npm run build:embedded
```

The embedded build writes its output to:

- `dashboard/frontend/dist/embedded/wizmap`

That matches the dashboard backend static lookup contract and the packaged container layout.

## Runtime contract

The hosted app reads these query parameters:

- `title`
- `dataURL`
- `gridURL`
- `topicURL`

The dashboard Knowledge Map page populates those values for one selected knowledge base and opens the hosted app inside `/knowledge-bases/:name/map`.

## Upstream origin

This shell is derived from the upstream `poloclub/wizmap` project and keeps the upstream `LICENSE` file in this directory.
