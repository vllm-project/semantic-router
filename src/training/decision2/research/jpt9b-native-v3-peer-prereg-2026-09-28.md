# JPT-9B native JevArena v3 same-panel peer: prospective execution lock

**Scope:** one fixed external comparison, not a Decision 2.0 candidate or an
official benchmark submission. The typed FINAL and CSS15 answer keys were
opened earlier for other project runs, so this is explicitly a **post-key
same-panel** comparison. This lock is written before this peer's FINAL/CSS15
predictions or score are produced. It cannot support a new blind-test claim.

## Fixed identity and inputs

| Item | Frozen value |
| --- | --- |
| Weights | `kirp/jpt-9b@7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e`, 9,409,813,744 stored parameters, six pinned configuration/tokenizer/shard hashes from prior native development manifest SHA-256 `fca2831ee09f315d4fd4f7c600fa6a4cd2c61e61b0da01757a16ed9277a1b33d` |
| Native inference | Published `llm2jev@2b252d504972764211ef172c1155ac0fedc9c3de`, `LLM2Jev(AutoProcessor, HF, temperature=1.087)`; original `state` and Choice/Noul/Score `questions`; no chat conversion, truncation, option deletion or relabeling |
| Adapter and loader | `inference/jpt.py` SHA-256 `1c1bfe9ced064e5a5175408dd76f317a8f7d88c810a8a765a37059053cdf90b4`; `inference/run.py` SHA-256 `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce` |
| Full-prediction seal | `jev_arena/seal_jpt9b_peer.py` SHA-256 `f71cd789b67b9bcbcacb7156ce818a35c14c64b3710ce0200986b502d4a00daf` |
| Typed FINAL | 1,600 gold-free items and 2,000 answer slots; prompt SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`; separate gold hash `707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e` |
| CSS15 | 6,547 gold-free items and answer slots across 15 human-label tasks; prompt SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`; separate gold hash `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4` |
| Scorers | Typed `benchmark/score.py` SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`; CSS `transfer/score.py` SHA-256 `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`; v3 aggregate `100 × sqrt(T × H)`, with four-family typed macro accuracy T and 15-task median CSS macro-F1 H |
| Fixed runtime | Offline ROCm image `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`, one live-verified idle MI325X GPU, BF16 published native backend |

The previously completed public JevBench 231 peer result is reused only if its
model revision, six model-file hashes, adapter, prompt SHA-256
`642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`,
public scorer SHA-256
`aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf`,
target SHA-256
`abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f`,
and sealed prediction SHA-256
`a8e7e2c75d3449fb1132821827fbf22a82b4852939d5585593b30aec0338e357`
still match. Do not rerun it merely to fill GPU capacity.

## Frozen execution and failure rule

1. Stage only exact local adapter/sealer/scorer bytes and gold-free prompts in
   a new private run directory; verify digests, model revision metadata,
   clean published `llm2jev` commit, source and image before GPU use. Keep
   gold outside every inference container. Use one isolated idle GPU; no
   existing experiment or cache is altered.
2. Repeat the earlier 32-row native smoke on that GPU. Stop on missing row,
   category change, invalid answer, model/source mismatch or maximum finite
   probability drift above 0.02. No formal gold is read here.
3. Execute complete typed FINAL and CSS15 gold-free panels **once each**, in
   that order, with the unchanged collector. A missing or invalid native
   answer is a failed answer under the existing scorer. If a panel aborts,
   OOMs or exceeds the bounded one-GPU-hour total/45-minute wall budget,
   retain the partial file and report incomplete; do not silently drop rows,
   truncate input or substitute another model/output path.
4. Validate all expected IDs, original question keys, input hashes and native
   manifest/model-file hashes. Hash, fsync and seal **both complete prediction
   files before reading either gold file**. Then score once using the fixed
   component scripts. Report T, H, v3 total, Choice/Noul/Score,
   four-family results, 15 task results, invalidity and probability quality.
   The prior public 231 tiers are separately labeled public replication.
5. Pair against own Lux 1.0 only if exact FINAL/CSS15 predictions with the
   same input and scorer digests are already available. Otherwise mark the
   paired 1.0 interval pending rather than mixing a DEV, pilot or Index row.
   No peer score is transferred to a Decision 2.0 model.

Record UTC start/end, one-device GPU-seconds, exact source/image/model/input/
prediction/report digests and failure details in the private receipt. The
public result note and unified research gist may contain only aggregate
statistics and non-sensitive digests, not raw prompts/predictions, protected
labels, private addresses, paths, credentials or logs. No HF publication.
