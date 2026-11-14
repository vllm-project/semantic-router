# Developing vSR

## Layout

| Path | What it holds |
| --- | --- |
| `.claude-plugin/plugin.json` | The manifest: name `vsr`, the `userConfig` options, the icon. |
| `.claude-plugin/marketplace.json` | A one-plugin marketplace (`vllm-sr`) for installing from this folder. |
| `hooks/register.ts` | The hooks module: every event is hooked here, and handed to one app module. |
| `hooks/router.ts`, `escalate.ts`, `output.ts`, `guard.ts`, `check.ts` | The apps. `output.ts` holds redact, shield and the write-back refusal. |
| `hooks/lib/shell.ts` | Bash read the way it runs: a command split into the commands it runs, and the fixed rules guard always holds. |
| `hooks/lib/settings.ts`, `hooks/lib/apps.ts` | Options as the engine hands them over (defaults live only in `plugin.json`), and each app's name and look. |
| `hooks/lib/state.ts`, `hooks/lib/log.ts`, `hooks/lib/labels.ts`, `hooks/lib/env.ts` | The session's decisions, the decision log, the answers given in `/vsr`, and where the log goes (`CLAUDE_CONFIG_DIR` aware). |
| `hooks/view.tsx`, `hooks/lib/draw.tsx`, `hooks/lib/describe.ts` | What each decision shows: rows, the band, the `/vsr` pane. `describe.ts` is the one table of words. |
| `hooks/lib/classify.ts` | The one way every app asks the runtime, with the offline circuit breaker. |
| `hooks/*.test.ts(x)`, `hooks/kit.tsx` | Tests, run by `claude plugin test .` against the engine; the kit stubs the runtime. |
| `skills/setup`, `skills/calibrate` | `/vsr:setup` and `/vsr:calibrate`, with their Python scripts and tests. `skills/setup/scripts/settings.py` reads option types and defaults from `plugin.json`; `framing.py` copies the classifier questions from the hooks, and its test fails when the two drift. The model runtime's `tests/test_claude_code_plugin_contract.py` sends each app's framed request to a tiny vela3-lite runtime and fails when the readers find no value. |
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
- The setup installs `vllm-srun` from a pinned commit of this repository (`COMMIT` in
  `skills/setup/scripts/local.py`, with that commit's `requirements-lock.txt` as constraints on Python
  3.12+); bump it with the runtime it was tested against.
