#!/bin/bash
# IB3 raw publisher files on node A (host side, network on): /data/dev2/private/data/ib3/raw (mode 700).
# HF repositories with the HF CLI at a pinned revision, GitHub files at a pinned commit, archives with curl.
# Every file's SHA-256 goes to download.log; the builder pins them. Only the files named here are kept.
# (Run on node A 2026-10-01 14:38Z; later edits are lint-only.)
set -euo pipefail
umask 077
R=/data/dev2/private/data/ib3/raw
[ ! -e "$R" ] || { echo "$R exists" >&2; exit 1; }
mkdir -p "$(dirname "$R")"
mkdir -m 700 "$R"
cd "$R"
export HF_HUB_CACHE=/data/dev2/hf-cache HF_HUB_DISABLE_TELEMETRY=1
log() { echo "$(date -u +%FT%TZ) $*" >> download.log; }
get() { # get <dir> <url> <name>
  mkdir -p "$1"
  chmod 700 "$1"
  curl -sSfL --retry 3 -o "$1/$3" "$2"
  log "curl $2 -> $1/$3 sha256=$(sha256sum "$1/$3" | cut -d' ' -f1)"
}

# Phishing URLs: PhiUSIIL (UCI 967, CC BY 4.0) and Web page phishing detection (Mendeley c2gw7fy2j4 v3, CC BY 4.0).
get uci_phiusiil https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip phiusiil+phishing+url+dataset.zip
get mendeley_wpd https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded dataset_B_05_2020.csv

# Grounding: HaluEval QA (MIT; GitHub RUCAIBox/HaluEval) and FaithDial train (MIT; HF McGill-NLP/FaithDial).
HALU=b7253db3cdaa0ab2c382f92b26b390109174f77e
for f in LICENSE README.md data/qa_data.json; do
  get "rucaibox_halueval/$(dirname "$f")" "https://raw.githubusercontent.com/RUCAIBox/HaluEval/$HALU/$f" "$(basename "$f")"
done
hf download McGill-NLP/FaithDial --repo-type dataset --revision 7a414e80725eac766f2602676dc8b39f80b061e4 \
  --include data/train.json --include README.md --local-dir mcgill_faithdial > /dev/null
rm -rf mcgill_faithdial/.cache
for f in mcgill_faithdial/data/train.json mcgill_faithdial/README.md; do log "hf McGill-NLP/FaithDial@7a414e80 $f sha256=$(sha256sum "$f" | cut -d' ' -f1)"; done

# Product search: Amazon Shopping Queries (ESCI; Apache-2.0), GitHub amazon-science/esci-data (LFS media).
ESCI=7916cdf6ab75a462e77f20ab40428a10923998d5
get amazon_esci "https://raw.githubusercontent.com/amazon-science/esci-data/$ESCI/LICENSE" LICENSE
for f in shopping_queries_dataset_examples.parquet shopping_queries_dataset_products.parquet; do
  get amazon_esci "https://media.githubusercontent.com/media/amazon-science/esci-data/$ESCI/shopping_queries_dataset/$f" "$f"
done

# Maths MCQ: MathQA (Apache-2.0), math-qa.github.io archive; only train.json is extracted.
get mathqa https://math-qa.github.io/math-QA/data/MathQA.zip MathQA.zip
python3 - <<'EOF'
import hashlib, zipfile
with zipfile.ZipFile("mathqa/MathQA.zip") as z:
    names = z.namelist()
    train = [n for n in names if n.rsplit("/", 1)[-1] == "train.json"]
    assert len(train) == 1, names
    data = z.read(train[0])
open("mathqa/train.json", "wb").write(data)
print("members", names)
EOF
log "extract mathqa/MathQA.zip train.json only -> mathqa/train.json sha256=$(sha256sum mathqa/train.json | cut -d' ' -f1)"

# Contracts: MAUD (CC BY 4.0), HF theatticusproject/maud, train CSV only.
hf download theatticusproject/maud --repo-type dataset --revision 37d5c3b95d18dcd8404cc5ce3fd5069be062392f \
  --include MAUD_v1/MAUD_train.csv --include README.md --local-dir theatticusproject_maud > /dev/null
rm -rf theatticusproject_maud/.cache
for f in theatticusproject_maud/MAUD_v1/MAUD_train.csv theatticusproject_maud/README.md; do log "hf theatticusproject/maud@37d5c3b9 $f sha256=$(sha256sum "$f" | cut -d' ' -f1)"; done
cp "$0" download.sh
log "done"
