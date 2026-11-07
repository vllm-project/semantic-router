# Developing vSR

## Layout

| Path | What it holds |
| --- | --- |
| `.claude-plugin/plugin.json` | The manifest: name `vsr`, the `userConfig` options, the icon. |
| `.claude-plugin/marketplace.json` | A one-plugin marketplace (`vllm-sr`) for installing from this folder. |
| `hooks/register.ts` | The hooks module: every event is hooked here, and handed to one app module. |
| `hooks/router.ts`, `escalate.ts`, `output.ts`, `guard.ts`, `check.ts` | The apps. `output.ts` holds redact, shield and the write-back refusal. |
| `hooks/view.tsx`, `hooks/lib/draw.tsx`, `hooks/lib/describe.ts` | What each decision shows: rows, the band, the `/vsr` pane. `describe.ts` is the one table of words. |
| `hooks/lib/classify.ts` | The one way every app asks the runtime, with the offline circuit breaker. |
| `hooks/*.test.ts(x)`, `hooks/kit.tsx` | Tests, run by `claude plugin test .` against the engine; the kit stubs the runtime. |
| `skills/setup`, `skills/calibrate` | `/vsr:setup` and `/vsr:calibrate`, with their Python scripts and tests. |
| `types/index.d.ts` | The decision types and the `$.state` contract. |
| `assets/` | The mark (`mark.png`), its 4-row block art for the demo (`mark.ansi`), the README screenshot. |
| `tools/mark_art.py` | Draws the mark in characters into `hooks/lib/mark-art.ts`. |

## Run and check

Start a runtime, then a session with the plugin loaded from this folder (it is keyed `vsr@inline`,
so its settings are separate from an installed copy's):

```bash
vllm-srun serve vllm-sr/vela3-lite --device cpu --host 127.0.0.1 --port 8790
claude --plugin-dir .
```

Before a change is done:

```bash
claude plugin validate .
claude plugin test .
python3 -m unittest discover -s skills/setup/scripts
python3 -m unittest discover -s skills/calibrate/scripts
```

## The logo characters

`assets/mark.png` is the source. `python3 tools/mark_art.py --preview out.png` (with Pillow and
numpy) regenerates `hooks/lib/mark-art.ts` and `assets/mark.ansi`, and saves a preview of both
character sets. The smooth set redraws the V on the grid of the Symbols for Legacy Computing wedges
first, so its edges stay straight.

## In this repository

- The root `.claude-plugin/marketplace.json` lists this folder as `vsr`, so users install it with
  `claude plugin marketplace add vllm-project/semantic-router` and `claude plugin install vsr@vllm-sr`.
- `tools/agent/domains.yaml` (`claude-code-plugin`) names the checks CI runs for this folder.
- The setup installs `vllm-srun` from a pinned commit of this repository
  (`skills/setup/scripts/local.py`); bump it with the runtime it was tested against.
