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
conventions (prereg amendment 2).
