# Exact research-source receipts

The `*.source` files preserve the exact Python bytes used by the authored-v3
candidate generator, multilingual translation smoke, and Laya r2 research
training/evaluation. `manifest.json` binds every file to its SHA-256 and
syntax-tree hash. The executable `.py` copies elsewhere in this branch were
formatted for repository checks; their parsed syntax trees match these frozen
sources exactly.

The authored-v3 dataset and Laya r2 weights both failed their release gates.
These snapshots are retained to explain and reproduce those experiments, not
to claim model qualification or editorial approval. Private seeds, question
targets, raw restricted training rows and experiment host details are absent.
