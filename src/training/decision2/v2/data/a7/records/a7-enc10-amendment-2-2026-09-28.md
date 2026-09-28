# A7 encoder corpora, amendment 2: SentiMix Hinglish joins A7s (2026-09-28)

Committed before any `a7-enc10-v3` row is built. Version `a7-enc10-v3` = the
rules of `a7-enc10-prereg-2026-09-28.md` and amendment 1, plus one source that
the preregistration deferred only because its file was missing on the node.
A7q, A7k and A7x must equal `a7-enc10-v2` row for row apart from the version
string in `audit_metadata.a7`; `a7-enc10-v2` is not published.

## Source

SentiMix Hinglish (SemEval-2020 Task 9, Patwa et al. 2020), used by the 1.0
encoder rosters (`sentimix-hinglish-valence`). Pinned file: the Zenodo release
`10.5281/zenodo.3974927`, `Semeval_2020_task9_data.zip`, SHA-256
`69509f1764ce24235fa2f64dec0b5f5e3f5ff7e9a484f3b6315264d8d978ec6d`, member
`Semeval_2020_task9_data/Hinglish/Hinglish_train_14k_split_conll.txt` (the
training split; the dev and test files are not read). Licence: CC BY 4.0 (the
Zenodo record's licence field). The Spanglish files in the same archive are not
used (not part of the 1.0 corpora).

## Rendering (family `sentimix_hinglish`, sub-arm A7s)

Each CoNLL block (`meta <id> <label>` followed by token/language-tag lines)
becomes one tweet: tokens joined by single spaces, language tags discarded.
Label negative / neutral / positive → Score level 0 / 1 / 2 with the A7s
sentiment template (`Message: …`), language code `hi-en`. Blocks with any other
label or no tokens are skipped; identical normalized texts form one group and
texts with conflicting labels are dropped. Balance, gates and publication are as
the preregistration (1.2 × rarest level within the family).
