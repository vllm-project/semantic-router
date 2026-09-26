# Exact research-source receipts

The `*.source` files preserve the exact Python bytes used by the authored-v3
and authored-v4 candidate generators, the A1 short-reasoning replay dataset
builder, multilingual translation smoke, and Laya r2 research
training/evaluation. `manifest.json` binds every file to its SHA-256 and
syntax-tree hash. The executable `.py` copies elsewhere in this branch were
formatted for repository checks; their parsed syntax trees match these frozen
sources exactly.

The authored-v3 and authored-v4 datasets and Laya r2 weights failed their
release gates. The A1 replay builder is a preregistered development control.
These snapshots are retained to explain and reproduce those experiments, not
to claim model qualification or editorial approval. Private seeds, question
targets, raw restricted training rows and experiment host details are absent.
