# JevArena-C1 policy P2: the class-aware PASS rule on item set v1.2, registered before the judgment and before any decryption (2026-09-29)

At 21:35 UTC+8 the coordinator adopted **P2**, the third option in the [event-3 record §000](m4-c1-event3-prep-2026-09-29.md).
Event 3 is scored on the registered item set **v1.2**, and the existing rescan hits are judged under a class-aware
PASS rule. This record registers that rule, the class map it uses and the re-pinned event script. It is committed and
pushed before the judgment runs and before the key is read. Nothing was decrypted: no C1 prompt, gold, salt,
prediction or key was read.

## 1. Item set

- **v1.2 as registered** ([amendment](m4-c1-v1_2-amendment-2026-09-29.md)): retired list `bbf095c7…`, 555 candidates,
  21 protected rows, at most 48 of 2,874 items. **No item is added.**
- **v1.3 is withdrawn and never used.** Its list `62fccb3d…` and its rescan verdict `c28843c4…` stay on record.
  `c1-rescan-coverage.json` marks it withdrawn, and `event3.sh` no longer names it.

## 2. The rule

The flagged rows are the merged non-CLEAN protected rows of the rescan. A flagged row is **still in the item set**
unless it is one of the 21 `protected_rows` of the v1.2 list. PASS requires, among the rows still in the item set:

1. **no OVERLAP or REVIEW hit in a class-(a) file**, the training rows of a released or candidate model;
2. **no near-exact hit in class (b)**, the other trained pools. Near-exact means containment ≥ 0.8, or an exact match
   of ≥ 8 tokens (the scanner's longest exact match in the file or group);
3. hits in classes **(c)**, **(d)** and **(e)** are counted and disclosed, **never failing**.

The judgment also fails on a coverage or binding problem:

- the grouped scan of a node misses a flagged row;
- the per-file scan misses a row with any hit under the training roots;
- a pinned class-(a) file is absent from every per-file manifest;
- the retired list, the protected rows or the node names don't match.

## 3. Classes (registered map [`c1-p2-classes.json`](../sealed/c1-p2-classes.json))

| Class | Registered definition |
| --- | --- |
| (a) training rows of a released or candidate model | Files whose SHA-256 is pinned, wherever the copy sits (the blob store included), plus the files under those mixtures' directories. The pins are the six released training files the program already pins (JevBench contamination record, `fresh_screen`): 0.6B T `98e4e859`, 0.8B E8F `d1dc33fc`, 2B S2T `13804ac6`, 4B N4XF `c7d51219`, 9B-tier K `a66131b1`, 27B F1 `de00df03`. Added: the six M6 mixtures of the 0.6B successor `m8-s5-b05` (the `m6-mxcx-soup` weights, approved 18:25, release in progress), pinned in `06b/m8_scorebias.py`. |
| (b) other trained pools | Every other group or file under the training roots: the training-data repository and its Hub delta, the blob store, `runs/` except `runs/eval`, `private/{data,27b,dec-arms,a7/runs,a7/base}`, `tmp/`. This is a superset of §000's (b): it also holds the CSS15 inventories, the audit cells and the blob store. |
| (d) evaluation-only pools | §000's pattern: panels, evals, gold-free and protected inventories, formal-v3, CSS transfer. |
| (c) raw source extracts | §000's pattern: `private/{sources,a7/sources,ml-sources}`, prior-session external datasets and source caches. |
| (e) other | Everything else, for example the raw-source sample dump under `private/tmp`. |

Two changes from §000:

- The 0.6B successor moves from (b) to (a). It is a candidate, even though event 3 doesn't score it.
- (b) is widened to the whole training root. That makes the near-exact test apply to more files, so the rule is
  stricter, not looser.

**Base-model training data.** Class (a) holds the program's own training rows of each model, the files its fine-tuning
read. Every program contamination check draws the line there too: the JevBench "released training files" and
successor-rule item 6. A base model's own training is outside it:

- Qwen pretraining;
- the Decision 1.0 corpora, which trained Sol, Nox, Eos and Lux 1.0 (the bases of DEV2.0-2B, 4B, 0.8B and 9B) and
  Kai and Lex 1.0.

C1's independence from these rests on its sources being published after the cutoffs, as for the peers. The registered
v1.2 coverage follows the same line: it scans the program's data roots, not the 1.0 projects' own roots.

Two parts of the 1.0 corpora do sit under the program's roots, so the rescan read them:

- `a7/sources/dec10`: pinned copies of the 1.0 decoder training files from which A7 was built;
- `a7/sources/enc10`: the raw upstream datasets behind the 1.0 encoder corpora.

They stay in (c), as in §000, and the verdict reports their hits separately.

**Known before registration.** A census of the existing hits was done to fix the class map (counts and paths only;
ids and text never printed). It found:

- **(a):** no row still in v1.2 has a hit in any class-(a) file, including the successor's.
- **(b):** 4 rows have REVIEW hits: 2 at containment ≤ 0.225 and 2 short exact matches of 6 tokens. Nothing is
  near-exact.
- **`dec10`:** 1 row, a tutoring transcript with 5,052 shingles. It has a 6-token exact match at containment 0 in
  the Stage-4 v2 curriculum (`063ac88e…`), which trained Sol and Nox 1.0. The same phrase is in A7m's held-out split,
  which no DEV2.0 model trained on.
- **`enc10`:** 18 rows hit the raw OASST1 file: 2 one-shingle DeliChess utterances at containment 1.0, and 16
  tutoring rows through 6-token phrases.
- **The (e) sample dump:** 2 OVERLAP rows, which also match raw sources. It holds the first rows of raw source
  snapshots (`make_samples.py`), and no pipeline reads it.

If (a) also counted a base model's own training rows, the `dec10` row alone would fail P2. The coordinator's
definition of (a) is the released and candidate *training mixtures* (21:35 note: "E8F, S2T, K"), and it is registered
as such.

## 4. How the rule was decided

- **Decided from content classes only.** The coordinator decided at 21:35 from the hit analysis in §000: counts by
  class, containment, matched corpora and item bounds per policy. No model output was used:
  - the key was never read and nothing was decrypted;
  - event 3 has no C1 prediction or score;
  - no event-1 or event-2 C1 result entered the decision.
- **Why.**
  - The independence claim concerns what the scored models trained on. The widened scan read every training file on
    both nodes, and it found only REVIEW-level matches in class (a). v1.2 retires all of them (6 rows).
  - The other matches are public text that C1 sources share with raw corpora no DEV2.0 model trained on (Tatoeba,
    Natural-Instructions, Taskmaster, MIRACL, ConvoKit, …).
  - Amendment 5 classes a match that traces to an independent upstream corpus as text-level overlap, not dataset
    presence. The seal record already discloses "label novelty, not text novelty".
  - P1 and P3 would retire 14–32% of C1 (up to both `tutormoments` tasks) for no independence gain.

## 5. The judgment: existing hits, no new scan

The inputs are pinned here. Ids and hits stay on node A.

| Input | Node A | Node B |
| --- | --- | --- |
| Rescan extract (v1.3 rescan `judge-20260929T125454Z`; the same 1,510 flagged rows as the v1.2 rescan) | `2df8c6d0…` (hits `1e6feabd…`) | `6f389628…` (hits `f955a0ee…`) |
| Grouped analysis hits (`c1-hitgroups/<node>/overlap-hits.jsonl`) | `4f39587d…` | `ae9efb32…` |
| Per-file hits of the 17 training-root rows | `0432e52d…` | `9ea9c10a…` |
| Per-file manifest | `03665804…` | `3ce68ba7…` |

- **Other inputs:** retired list `bbf095c7…`, protected rows `36797f50…`, class map `59c180b5…`.
- **Command** (node A, from the mirror of the registering commit, CPU only):
  `python3 -m v2.eval.sealed.scanverdict judge-classes --extract … --grouped nodeA=… --grouped nodeB=… --perfile … --perfile-manifest … --classes v2/eval/sealed/c1-p2-classes.json --retired … --retired-sha bbf095c7… --protected-sha 36797f50… --output /data/dev2/runs/eval/m4/c1-rescan-v1_2/SCAN-VERDICT-P2.json`.
- **Recording.** The verdict (PASS or FAIL) and its SHA-256 are recorded in a follow-up commit before the key is
  read. On FAIL the event stops and stays unused.

## 6. What changes, and what doesn't

- **`event3.sh`** pins item set v1.2:
  - `RETIRED_SHA` = `bbf095c7…`;
  - `SCAN_VERDICT` = `c1-rescan-v1_2/SCAN-VERDICT-P2.json`;
  - `check-v2 --schema dev2-c1-class-verdict/1`, with the file pinned by `--scan-verdict-sha`.
- **Scoring.** Every score and pairing drops v1.2's retired candidates. That includes Kai 1.0 at 8K against the
  stored DEV2.0-0.6B event-2 predictions.
- **Unchanged:** the batch, every smoke before the key, the stored-seal checks, and the procedure in the event-3
  record §5–§8.
- **GPU:** node A GPU0 or GPU1, whichever has ≥ 130 GB free, under a shared lease with the entry `owner.eval`.
- **Code:** `scanverdict judge-classes` and `check-v2 --schema` (tests in `tests/test_scanverdict.py`, including
  that the class-(a) pins equal the release and successor records), and `tests/test_rescan.py`.
