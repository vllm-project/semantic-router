"""Census model-visible script and optional language-ID evidence without emitting text.

Only local development panels are accepted by the caller. This module never
reads sealed final data, writes rows, or includes any source text in its report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)


def visible_text(row: dict, kind: str) -> tuple[str, str]:
    if kind == "training":
        fields = [row.get("instructions"), row.get("state"), row.get("options")]
        main = row.get("state")
    elif kind == "prompts":
        fields = [row.get("state"), row.get("questions")]
        main = row.get("state")
    else:
        raise ValueError(f"Unknown row kind {kind!r}")
    return "\n".join(s for field in fields for s in strings(field)), "\n".join(
        strings(main)
    )


def script_of(char: str) -> str | None:
    if not char.isalpha():
        return None
    point = ord(char)
    if 0x4E00 <= point <= 0x9FFF or 0x3400 <= point <= 0x4DBF:
        return "han"
    if 0x3040 <= point <= 0x30FF or 0x31F0 <= point <= 0x31FF:
        return "kana"
    if (
        0x0600 <= point <= 0x06FF
        or 0x0750 <= point <= 0x077F
        or 0x08A0 <= point <= 0x08FF
    ):
        return "arabic"
    if 0x0400 <= point <= 0x052F:
        return "cyrillic"
    if 0x0900 <= point <= 0x097F:
        return "devanagari"
    if 0xAC00 <= point <= 0xD7AF or 0x1100 <= point <= 0x11FF:
        return "hangul"
    if 0x0370 <= point <= 0x03FF:
        return "greek"
    name = unicodedata.name(char, "")
    return "latin" if "LATIN" in name or point < 128 else "other"


def evidence(text: str, state_text: str, *, langid=None) -> dict:
    scripts = Counter(filter(None, (script_of(char) for char in text)))
    letters = sum(scripts.values())
    nonlatin = sum(n for name, n in scripts.items() if name != "latin")
    significant_nonlatin = nonlatin >= 5 and nonlatin >= 0.05 * max(1, letters)
    state_letters = sum(char.isalpha() for char in state_text)
    result = {
        "scripts": scripts,
        "letters": letters,
        "nonlatin_letters": nonlatin,
        "significant_nonlatin": significant_nonlatin,
        "state_letters": state_letters,
    }
    if langid is not None and letters >= 60:
        ranked = langid.rank(text)
        language, first = ranked[0]
        margin = float(first - ranked[1][1])
        result.update(
            {
                "langid": language,
                "langid_margin": margin,
                "langid_confident": margin >= 5,
            }
        )
        if state_letters >= 60:
            state_ranked = langid.rank(state_text)
            result["state_langid"] = state_ranked[0][0]
    return result


def read_metadata(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError("Duplicate metadata ID")
            result[row["id"]] = {
                k: row.get(k)
                for k in ("family", "source", "task", "category", "tier", "modality")
            }
    return result


def source_of(row: dict, metadata: dict, source_key: str) -> str:
    if source_key == "css_id":
        return row["id"].split("/")[1]
    if source_key == "mldev_language":
        language = row["id"].split("-")[-2]
        if language not in {"en", "zh", "es", "fr", "de", "ja", "ar"}:
            raise ValueError("Unexpected multilingual pilot ID")
        return language
    if source_key == "parallel_language":
        language = row["id"].split("-")[-1]
        if language not in {"en", "zh", "es", "fr", "de", "ja", "ar"}:
            raise ValueError("Unexpected multilingual parallel ID")
        return language
    if source_key == "metadata_family":
        return str(metadata.get("family") or "unknown")
    if source_key == "metadata_category":
        return str(metadata.get("category") or "unknown")
    if source_key == "metadata_task":
        return str(metadata.get("task") or "unknown")
    return str(row.get(source_key) or metadata.get(source_key) or "unknown")


def audit(
    path: Path,
    *,
    kind: str,
    source_key: str,
    metadata_path: Path | None = None,
    langid=None,
) -> dict:
    metadata = read_metadata(metadata_path)
    counts = Counter()
    scripts = Counter()
    declared = Counter()
    langid_all = Counter()
    langid_confident = Counter()
    sources = defaultdict(Counter)
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row_id = row["id"]
            if row_id in seen:
                raise ValueError("Duplicate panel ID")
            seen.add(row_id)
            text, state = visible_text(row, kind)
            e = evidence(text, state, langid=langid)
            source = source_of(row, metadata.get(row_id, {}), source_key)
            bucket = sources[source]
            counts["rows"] += 1
            bucket["rows"] += 1
            counts["visible_letters"] += e["letters"]
            counts["nonlatin_letters"] += e["nonlatin_letters"]
            scripts.update(e["scripts"])
            bucket["visible_letters"] += e["letters"]
            bucket["nonlatin_letters"] += e["nonlatin_letters"]
            if e["significant_nonlatin"]:
                counts["significant_nonlatin_rows"] += 1
                bucket["significant_nonlatin_rows"] += 1
            if e["letters"] < 60:
                counts["short_lt60_letters"] += 1
            if e["state_letters"] < 60:
                counts["state_short_lt60_letters"] += 1
            language = row.get("language")
            if language is not None:
                declared[language] += 1
                bucket[f"declared_language_{language}"] += 1
            if "langid" in e:
                langid_all[e["langid"]] += 1
                bucket[f"langid_top_{e['langid']}"] += 1
                if e["langid_confident"]:
                    langid_confident[e["langid"]] += 1
                    if e["langid"] != "en":
                        bucket["confident_langid_non_en"] += 1
                else:
                    counts["langid_margin_lt5"] += 1
                if "state_langid" in e and e["state_langid"] != e["langid"]:
                    counts["state_vs_full_langid_disagreement"] += 1
    if metadata and seen != set(metadata):
        raise ValueError("Prompt/metadata IDs differ")
    return {
        "schema_version": "decision2-multilingual-language-audit/1",
        "scope": "public development and training files only; no source text or sealed final",
        "path_name": path.name,
        "input_sha256": sha256(path),
        "metadata_sha256": sha256(metadata_path) if metadata_path else None,
        "kind": kind,
        "source_key": source_key,
        "langid_version": "1.1.6" if langid is not None else None,
        "langid_policy": "full visible strings; >=60 Unicode letters; confident if top-two log-score margin >=5",
        "script_policy": "significant non-Latin if >=5 and >=5% of alphabetic characters",
        "counts": dict(counts),
        "scripts": dict(scripts),
        "declared_language": dict(declared),
        "langid_top": dict(langid_all),
        "langid_confident": dict(langid_confident),
        "by_source": {key: dict(value) for key, value in sorted(sources.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("training", "prompts"), required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--langid", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    detector = None
    if args.langid:
        import langid

        detector = langid
    report = audit(
        args.input,
        kind=args.kind,
        source_key=args.source_key,
        metadata_path=args.metadata,
        langid=detector,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": report["counts"]["rows"],
                "input_sha256": report["input_sha256"],
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
