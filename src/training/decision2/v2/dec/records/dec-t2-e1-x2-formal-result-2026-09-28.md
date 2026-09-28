# E1-X2 (4B) post-key same-panel result: HOLD

Candidate, runner, panels and reading were fixed by the
[lock](dec-t2-e1-x2-formal-lock-2026-09-28.md) (`f13c03006`) and its
[same-node amendment](dec-t2-e1-x2-formal-lock-amendment-1-2026-09-28.md)
(`b91181d0e`) before any prediction was scored. The v3 labels were accessed
earlier in the project: this is a **post-key same-panel** comparison, not a new
blind test. JevBench public 231 is a public-subset reproduction, not the official
sealed rank.

- Collection: node A GPU5, `v2/eval/run_same_panel.sh` from exact mirror
  `8e5f22cd8`, image `sha256:f83b1d10…`, `TRITON_CACHE_AUTOTUNING=1` with a
  run-specific cache; adapter `v2/dec/adapter-spec-infer-dec.json`; collected
  identity `de41a699e3a3…` (= lock), CAL temperatures as locked; 1,600 + 6,547 +
  231 rows, 0 typed and 0 public invalid, 18 CSS15 over-budget (`tropes`).
  Seal `034192be53a1cc12f6d74e83200f2f133e46334cd95817387f23b56e8bda813c`.
  A first full launch was refused by the runner's idle check (VRAM 2% while the
  smoke container released memory) before any collection; the relaunch used the
  same empty run directory.
- Comparator: eval-track adopted Nox 1.0 run `m1-adopt/nox1` (node A).

| Post-key same-panel | E1-X2 | Nox 1.0 |
| --- | ---: | ---: |
| JevArena v3 | 55.993 | 56.470 |
| T / H | .5844 / .5365 | .6144 / .5190 |
| Choice / Noul / Score | 551 / 616 / 168 | 552 / 653 / 178 |
| Families: constraint / evidence / exception / ledger | 151 / 800 / 216 / 168 | 152 / 800 / 253 / 178 |
| Public 231 (E/S/H) | 174 (48/67/59) | 173 (48/66/59) |
| Typed Brier / ECE10 | .232 / .140 | .205 / .092 |

Paired v3 difference −0.477, 95% CI [−3.090, +1.780]. Reading fixed in the lock
(v3 ≥ +3.0 with lower bound > 0 and public231 ≥ 173): **HOLD**. No HF upload,
no second 4B candidate on v3 in Milestone 1, no change to the package.

Cross-node repeat (node B GPU0, image `ce895822…`, seal `dfb39739…`): 56 of
8,778 answers differ from the node A collection (typed 22, CSS15 32, public 2);
not scored, per the amendment.
