"""Issue an exact, reviewable rights statement for the frozen 5824 research arm.

The statement records upstream conditions and unresolved corpus terms. It is
limited to noncommercial research weights and a model card; no source rows are
included. It does not turn an upstream research permission into a new license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "decision2-noncommercial-research-attestation/1"
SCOPE = "noncommercial research model weights/modelcard only; no raw rows"
MANIFEST_SCHEMAS = {
    "decision2-balanced-human-5824/1",
    "decision2-nox4b-structured-replay/1",
}
HOLDOUT_COUNTS = {
    "select": {
        "css_pilot:semeval_stance": 200,
        "css_pilot:implicit_hate": 200,
        "css_pilot:discourse": 200,
    },
    "cal": {
        "css_pilot:semeval_stance": 100,
        "css_pilot:implicit_hate": 100,
        "css_pilot:discourse": 100,
        "decision2_cal_hard_original_v2": 600,
    },
}
SOURCES = {
    "css_flute_official_train": "flute",
    "decision2_programmatic_original_v1": "internal",
    "decision2_targeted_programmatic_v1": "internal",
    "legacy:cosmos_qa": "cosmos_qa",
    "legacy:nyu-mll/multi_nli": "multi_nli",
    "legacy:snli": "snli",
    "legacy:squad2_answerability": "squad2",
    "legacy:stage3_replay": "stage3_mixed",
    "legacy:stage4-general-composition-v2": "internal",
    "tweeteval_train:emotion": "nrc_emotion",
    "tweeteval_train:hate": "hateval",
    "tweeteval_train:irony": "semeval_irony",
    "tweeteval_train:offensive": "olid",
    "tweeteval_train:sentiment": "semeval_sentiment",
    **{
        f"tweeteval_train:stance/{topic}": "nrc_stance"
        for topic in ("abortion", "atheism", "climate", "feminist", "hillary")
    },
}
HOLDOUTS = {
    "css_pilot:semeval_stance": "nrc_stance",
    "css_pilot:implicit_hate": "implicit_hate",
    "css_pilot:discourse": "coarse_discourse",
    "decision2_cal_hard_original_v2": "internal",
}
CONDITIONS = {
    "internal": {
        "terms": "Objective programmatic examples generated for Decision 2.0 research; retain generator provenance; no upstream text license.",
        "evidence": "https://github.com/vllm-project/semantic-router",
    },
    "stage3_mixed": {
        "terms": "Mixed replay: internally generated cases plus CLINC150 CC BY 3.0 and BANKING77 CC BY 4.0 cases. Attribute original sources; do not distribute raw rows in the model package.",
        "evidence": "https://github.com/clinc/oos-eval/blob/master/LICENSE ; https://huggingface.co/datasets/PolyAI/banking77",
    },
    "flute": {
        "terms": "FLUTE official TRAIN, AFL 3.0; attribute ColumbiaNLP and disclose same-task CSS exposure. No raw rows in model package.",
        "evidence": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE",
    },
    "cosmos_qa": {
        "terms": "Cosmos QA official TRAIN, CC BY 4.0; attribute AllenAI. No raw rows in model package.",
        "evidence": "https://huggingface.co/datasets/allenai/cosmos_qa",
    },
    "multi_nli": {
        "terms": "MultiNLI TRAIN origin; repository code MIT and OANC underlying genres include permissive material, but annotation and derivative-weight terms have not been fully established. Noncommercial research disclosure only; no raw rows.",
        "evidence": "https://github.com/nyu-mll/multiNLI ; https://www.anc.org/data/oanc/",  # codespell:ignore anc
    },
    "snli": {
        "terms": "SNLI official TRAIN, CC BY-SA 4.0; attribute Stanford and disclose share-alike source. No raw rows in model package.",
        "evidence": "https://nlp.stanford.edu/projects/snli/",
    },
    "squad2": {
        "terms": "SQuAD 2.0 official TRAIN, CC BY-SA 4.0; attribute Stanford and disclose share-alike source. No raw rows in model package.",
        "evidence": "https://rajpurkar.github.io/SQuAD-explorer/",
    },
    "nrc_emotion": {
        "terms": "NRC SemEval emotion annotations are available for research; creator requires separate arrangements for commercial use and restricts third-party text redistribution. Noncommercial research weights/card only; no raw tweets.",
        "evidence": "https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html",
    },
    "nrc_stance": {
        "terms": "NRC SemEval stance annotations are available for research; creator requires separate arrangements for commercial use and restricts third-party text redistribution. Noncommercial research weights/card only; no raw tweets.",
        "evidence": "https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html",
    },
    "hateval": {
        "terms": "HatEval original task data are CC BY-NC 4.0; retain attribution, noncommercial use only, no raw tweets in public model package.",
        "evidence": "https://hatespeech.di.unito.it/hateval.html",
    },
    "semeval_irony": {
        "terms": "SemEval 2018 Task 3 irony data specify CC BY-NC-SA 4.0 and academic research purpose; attribute task organizers and do not redistribute raw tweets.",
        "evidence": "https://github.com/Cyvhee/SemEval2018-Task3",
    },
    "olid": {
        "terms": "OLID/OffensEval public research task data; an affirmative commercial corpus license was not located. Noncommercial research weights/card only; no raw tweets.",
        "evidence": "https://sites.google.com/site/offensevalsharedtask/olid",
    },
    "semeval_sentiment": {
        "terms": "SemEval 2017 Task 4 public research task data; an affirmative commercial corpus license was not located. Noncommercial research weights/card only; no raw tweets.",
        "evidence": "https://alt.qcri.org/semeval2017/task4/",
    },
    "implicit_hate": {
        "terms": "SALT implicit-hate repository carries MIT code license but requests a survey before corpus access; this does not establish rights in copied CSS text. Noncommercial research holdout use; terms unresolved and no raw rows redistributed.",
        "evidence": "https://github.com/SALT-NLP/implicit-hate",
    },
    "coarse_discourse": {
        "terms": "Original coarse-discourse release distributes Reddit metadata without comment text, obtained separately via API. Rights in copied CSS text are unresolved; noncommercial research holdout use only and no raw rows redistributed.",
        "evidence": "https://github.com/google-research-datasets/coarse-discourse",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest_path: Path, provenance_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in MANIFEST_SCHEMAS:
        raise ValueError(
            "Expected a frozen balanced-human or structured-replay data manifest"
        )
    sources = data.get("counts", {}).get("source")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("Frozen TRAIN source roster changed")
    data_sha = (
        provenance.get("contract", {}).get("data_sha256")
        if isinstance(provenance.get("contract"), dict)
        else provenance.get("data_sha256")
    )
    if not isinstance(data_sha, dict) or set(data_sha) not in (
        {"train", "select", "cal_audited_only"},
        {"train", "select", "cal"},
    ):
        raise ValueError("Training provenance lacks exact pilot split hashes")
    outputs = data.get("outputs", {})
    train_name = (
        "balanced_human_5824.train.jsonl"
        if data["schema_version"] == "decision2-balanced-human-5824/1"
        else "nox4b_structured.train.jsonl"
    )
    cal_role = "cal" if "cal" in data_sha else "cal_audited_only"
    for role, name in (
        ("train", train_name),
        ("select", "select.jsonl"),
        (cal_role, "cal.jsonl"),
    ):
        if outputs.get(name, {}).get("sha256") != data_sha[role]:
            raise ValueError(f"Frozen {role} output differs from run provenance")
    groups = {
        role: {name: HOLDOUTS[name] for name in counts}
        for role, counts in HOLDOUT_COUNTS.items()
    }
    if set(SOURCES.values()).union(HOLDOUTS.values()) != set(CONDITIONS):
        raise AssertionError("Rights condition roster differs from mapped sources")
    return {
        "schema_version": SCHEMA,
        "noncommercial_use": True,
        "publication_scope": SCOPE,
        "no_raw_training_rows": True,
        "data_manifest_sha256": _sha(manifest_path),
        "training_provenance_sha256": _sha(provenance_path),
        "data_sha256": data_sha,
        "source_counts": sources,
        "source_groups": SOURCES,
        "holdout_source_counts": HOLDOUT_COUNTS,
        "holdout_groups": groups,
        "rights_conditions": CONDITIONS,
        "limitations": [
            "This records provenance and source conditions; it is not a new license or a determination of third-party derivative-weight rights.",
            "SALT implicit-hate and coarse-discourse underlying text terms remain unresolved.",
            "MultiNLI annotation and derivative-weight terms remain unresolved.",
            "TweetEval source-task terms vary; research-only and NC conditions are retained.",
            "FLUTE TRAIN exposure makes CSS FLUTE same-task supervised.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    statement = build(args.manifest, args.provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(statement, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "sha256": _sha(args.output),
                "sources": len(SOURCES),
                "holdouts": sum(map(len, HOLDOUT_COUNTS.values())),
            }
        )
    )


if __name__ == "__main__":
    main()
