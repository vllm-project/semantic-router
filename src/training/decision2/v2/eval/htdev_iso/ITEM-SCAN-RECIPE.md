# HT-DEV item-level scans for the builder (prereg §3 items 4 and 5(i))

Run on node A from an exact code mirror; every Python step in `decision20-train-fast:host2` with
`--network none`. `POOL` is the builder's candidate pool in the C1 candidate shape (`overlap_texts`
plus `state` leaves; one line per item). Outputs are written O_EXCL, mode 600; only counts leave the
scanner. Never read `/data/dev2/private/sealed/`, never glob `/data/dev2/private/panels/goldfree/`.

```bash
SRC=/data/dev2/src/<pushed commit>-src_training_decision2; S=$SRC/src/training/decision2
ISO=/data/dev2/private/htdev/iso; G=/data/dev2/private/panels/goldfree
POOL=/data/dev2/private/htdev/<builder dir>/pool.jsonl; W=/data/dev2/private/htdev/<builder dir>/scans
RUN="docker run --rm --network none -v $SRC:$SRC:ro -v /data/dev2/private/htdev:/data/dev2/private/htdev \
  -v /data/dev2/private/eval:/data/dev2/private/eval:ro -v /data/dev2/private/data:/data/dev2/private/data:ro \
  -v /data/dev2/private/panels/goldfree:$G:ro -v /data/dev2/runs:/data/dev2/runs:ro -v /data/dev2/src:/data/dev2/src:ro \
  -e PYTHONPATH=$S -e PYTHONDONTWRITEBYTECODE=1 -w $S decision20-train-fast:host2"
umask 077; mkdir -p $W

# 0. drop the source rows ADMISSION.json flags (match any leaf_sha256 against
#    sha256(v2.eval.sealed.schema.normalized(text)) of the item's source-row string leaves >= 20 chars)

# (a) item-level lexical scan against every training corpus of the frozen manifest
$RUN python3 -m v2.eval.sealed.overlap scan --protected $POOL \
  --manifest $ISO/training-corpora.json --workers 48 --exact-min-tokens 8 \
  --output $W/items-train-receipt.json --hits $W/items-train-hits.jsonl

# (b) item-level lexical scan against the protected eval panels (six gold-free files, named)
$RUN python3 -m v2.eval.sealed.overlap scan --protected $POOL \
  --corpus panel-typed-final=$G/typed-final.prompts.jsonl --corpus panel-css15=$G/css15.prompts.jsonl \
  --corpus panel-public231=$G/public231.prompts.jsonl --corpus panel-typed-dev=$G/typed-dev.prompts.jsonl \
  --corpus panel-css-pilot=$G/css-pilot.prompts.jsonl --corpus panel-mlx-diag=$G/mlx-diag.prompts.jsonl \
  --workers 48 --exact-min-tokens 8 --output $W/items-panels-receipt.json --hits $W/items-panels-hits.jsonl
```

Drop rule (prereg §3.4): drop every item whose verdict in (a) or (b) is OVERLAP (exact match or
containment >= 0.5) or REVIEW (containment 0.2-0.5, or an exact match shorter than 8 tokens), and
items with 0 shingles. Check first that `training_manifest_sha256` in ADMISSION.json equals
`sha256sum $ISO/TRAINING-MANIFEST.json`; the scanner verifies every corpus file's sha256 and size
against `training-corpora.json` (do not pass `--no-verify`).

Embedding (prereg §3.5(i)): export `{id, group_id, state}` rows of the pool and run
`python3 -m v2.data.embed_scan --candidates <pool rows> --protected-inventory <PI>` with
Qwen3-Embedding-0.6B @97b0c614 (`/data/dev2/hf-cache/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`),
`--batch 256 --quarantine 0.93 --review 0.85 --sample 20 --seed decision2-embed-scan-v1`, once with the
PI-v4 panel inventory (`/data/dev2/runs/data/m3b/gap/c1/pi/manifest.json`) and once with the
training-social inventory `$ISO/work/embed/pi/manifest.json`; drop items at >= 0.93. GPU job under a
shared lease (see `v2/eval/htdev_iso/run_embed.sh` for the docker flags).
