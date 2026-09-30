# Decision 2.0 9B + CLM track

Frozen-backbone feature extraction, CLM-style dual-projection readouts and
ordinary Decision-head controls for the 9B tier. Records (protocols,
preregistrations, results) are in `records/`.

Run modules from `src/training/decision2` with this directory on the path:

```bash
cd src/training/decision2
PYTHONPATH=.:v2/9b python -m clm9b.extract --help
python -m unittest discover -s v2/9b/tests -p 'test_*.py'
```

Remote nodes run only exact mirrors of pushed commits inside the pinned
trainer image; see `records/` for the frozen commands.

Milestone 3 (Lux 1.0 continuation on data v2 / A7): `lux9b/m3_data.py`
materializes a TRAIN partition and its teacher file from a frozen spec in
`lux9b/specs/` (hash-verified inputs, A0 dedupe, token budgets, sealed-C1 and
mlx-diag source guard, isolation). Node-side wrappers in `lux9b/m3/`:
`data.sh` (CPU build), `arm.sh` (preflights, training, CAL698, development
readouts), `soup.sh` (seed soups and interpolations), `score.sh` (development
readout), `formal.sh` (16K post-key runs against the Lux1 comparators), and the
`wave*.sh` drivers that launched each wave.

Milestone 4 (arm D's full fine-tuning on the XL r2 recipe): the same builder
takes `recipe_exclude_pools` and `recipe_budget_tokens` (whole groups,
stratified by pool x source x task type x language, seed `<seed>:recipe`,
strict own-Lux coverage of every selected row). Specs
`m4-k-xl-r2-60m.json` (all pools) and `m4-kn-xl-r2-nohum-60m.json` (without
A7q / H1 / H8), 60M native tokens each. Node-side wrappers in `lux9b/m4/`
(run root `/data/dev2/runs/9b/m4`, the M3 root mounted read-only as `/m3`,
own training autotune cache copied once from M3): `data.sh`, `arm.sh`
(`--kl W` strict teacher or `--no-teacher`), `readout.sh` / `soup.sh` /
`interp.sh` / `lux_readout.sh` (CAL698 + development predictions at 16,384
tokens; soups accept repeated members for rational weights), `score.sh` and
`formal.sh` (runs under `formal-m4`, one copy of the frozen `formal-m3` cache).
GPU time: `python3 lux9b/m3/gpu_hours.py /data/dev2/runs/9b/m4 /data/dev2/runs/9b/formal-m4`.
`chain-step.sh` runs one hand-launched chain step (node B GPU0–2, or a node-A
GPU whose chain driver was stopped) with the chain drivers' lease and log
conventions (prereg amendment 2). `lux9b/m4_rules.py` (`seed` / `alpha`,
tests `tests/test_m4_rules.py`) applies the preregistered seed, alpha and
proxy-drop rules to the development readouts, with exact rational floors
(prereg amendment 3).

Milestone 5: `lux9b/m5_rules.py` (`seed` / `alpha` / `finalists`) applies the
incumbent-anchored alpha rule (reference = the M5 re-read of K-a13, line-local
proxy drop, M4's Lux-anchored pick reported alongside). Two post-key CPU
reporters run from a mirror with
`PYTHONPATH=<mirror>/src/training/decision2:<mirror>/src/training/decision2/v2/9b`:
`lux9b/mlx_paired.py` (paired, type x language stratified bootstrap of the
card-eligible mlx-diag Choice + Noul parts, self-checked against both stored
`mlx-diag.score.json`) and `lux9b/score_levels.py` (typed FINAL Score level
usage via `v2.eval.gates.type_summary`).

Milestone 6: `lux9b/m6_data.py` (CPU; `split` / `ka` / `kh`) builds the
human-rated soft-target set S on the K recipe (x60) and the AutoJev-27B wave
inputs for S rows without production targets, the KA teacher file (AutoJev on
S, own-Lux elsewhere; train.jsonl is x60 byte for byte), and the KH TRAIN (x60
cut in stratified whole groups by the tokens of an HS1 block, plus the block,
gold only). `lux9b/m6_rules.py` (`seed` / `early` / `alpha` / `finalists`)
adds a Noul `rule_precedence` floor to M5's incumbent-anchored alpha rule and
the early stop at an arm's first full checkpoint. Node wrappers in `lux9b/m6/`
follow M5's (`aj_job.sh` / `aj_wave.sh` run the qualified AutoJev collector on
GPU6-7; `early.sh`, `hs1.sh`; `formal.sh` adds the public-231 gate, successor
item 7).

Milestone 7: `lux9b/m7_data.py` (CPU; `topup` / `mlxdev`) builds the two continuation TRAIN
files on the K recipe (P = x60 replay + the PN1-r2 block, gold only on the block; C = x60 replay
at P's native tokens, containing P's replay) and MLX-DEV-9B (the decoder's MLX-DEV panel minus
every group that shares a group, id, input or segment with x60 or PN1-r2). `lux9b/m7_rules.py`
(`pn1` / `early` / `alpha` / `finalists`) scores the PN1 dev slice (hop and clean gold-no
yes-rates), applies the member-1 early stop and M6's alpha rule with the CSS-pilot H3 condition
replaced by the HT-DEV v2, PN1 dev and MLX-DEV-9B screens. Node wrappers in `lux9b/m7/`
follow M6's; `cont.sh` continues a K seed with `v2.dec.train_dec --init decision2`, `line.sh`
builds a soup and its interpolations and reads them, `screens.sh` scores the screens;
`dryrun_test.sh` exercises them locally.
