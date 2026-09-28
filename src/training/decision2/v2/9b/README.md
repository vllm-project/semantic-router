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
