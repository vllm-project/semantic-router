# Milestone 2 operational wrappers (verbatim records)

Exact copies of the shell/Python wrappers that were executed on the nodes
during decoder Milestone 2, kept as text so the sequence of jobs is
reviewable. Every job they started also left a launch receipt (full docker
argv, image, commit, start/end UTC, exit status) or, for formal runs, the eval
runner's `COLLECT.json` / `GPU-TIME.json` beside its output on the node. Node
aliases resolve from a private file; no address or credential appears here.
