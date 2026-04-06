# Deep Dive into vLLM Semantic Router

This folder contains the expanded long-form presentation package for a `40` to
`60` minute technical deep dive.

The visual direction stays white-background / black-text / OpenAI-like, but the
content is now organized as a production-style talk rather than a short
overview.

## Files

- `deck_brief.md` — scope, audience, timing, visual lock, and narrative contract
- `deck_script.md` — 27-slide long-form script with timing and source anchors
- `speaker_notes_long.md` — expanded presenter notes for the full talk
- `source_map.md` — slide-to-source evidence map
- `demo_runbook.md` — live-demo path, preflight, and fallback plan
- `index.html` — main 27-slide deck, self-contained and browser-openable
- `appendix.html` — backup deck for Q&A and deeper technical detail

## Recommended Use

- Use `index.html` for the main `45` to `55` minute presentation.
- Use `appendix.html` for Q&A, backup slides, or a longer workshop-style session.
- Keep `speaker_notes_long.md` open during rehearsal.
- Use `source_map.md` when you want to verify a claim or update the deck later.

## Open It

Open the HTML files directly in a browser, or serve the folder locally:

```bash
cd docs/presentations/deep-dive-vllm-semantic-router
python3 -m http.server 9000
```

Then visit:

- `http://localhost:9000/index.html`
- `http://localhost:9000/appendix.html`

## Controls

- `Right Arrow`, `Space`, `PageDown` — next slide
- `Left Arrow`, `PageUp` — previous slide
- `Home` — first slide
- `End` — last slide

## Main Source Anchors

The long-form deck is grounded in repository-native sources, especially:

- `website/docs/intro.md`
- `website/docs/overview/goals.md`
- `website/docs/overview/semantic-router-overview.md`
- `website/docs/overview/collective-intelligence.md`
- `website/docs/tutorials/signal/overview.md`
- `website/docs/tutorials/projection/overview.md`
- `website/docs/tutorials/plugin/overview.md`
- `website/docs/tutorials/algorithm/overview.md`
- `website/docs/installation/configuration.md`
- `website/docs/training/training-overview.md`
- `website/docs/fleet-sim/overview.md`
- `website/docs/fleet-sim/dashboard-integration.md`
- `src/vllm-sr/README.md`
- `dashboard/README.md`
- `deploy/operator/README.md`
- `deploy/amd/README.md`
- `docs/agent/repo-map.md`
- `docs/agent/context-management.md`
- `docs/agent/architecture-guardrails.md`

## Note

Current docs are slightly inconsistent about signal count. This package uses the
current maintained count of `16` signal families from `website/docs/intro.md`
and `website/docs/tutorials/signal/overview.md`, instead of the older `15`
count still present in `website/docs/overview/signal-driven-decisions.md`.
