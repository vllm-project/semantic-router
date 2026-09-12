# Fullwidth punctuation is intentional in Chinese and Japanese training text.
# ruff: noqa: RUF001
"""Build a frozen PII repair corpus from MIT Presidio and original templates.

Generated examples are synthetic. They measure the declared formats and
contexts, not production accuracy. Existing runtime diagnostics are not inputs.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from span_data import assert_split_isolation, load_label_contract, validated_spans

TYPES = [
    "AGE",
    "CREDIT_CARD",
    "DATE_TIME",
    "DOMAIN_NAME",
    "EMAIL_ADDRESS",
    "GPE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "NRP",
    "ORGANIZATION",
    "PERSON",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "TITLE",
    "US_DRIVER_LICENSE",
    "US_SSN",
    "ZIP_CODE",
]
FIELDS = {
    "en": "age|card number|date|website|email address|city|IBAN|IP address|nationality|organization|full name|phone number|street address|title|driver license|social security number|postal code",
    "zh": "年龄|银行卡号|日期|网站|电子邮箱|城市|国际银行账号|IP地址|国籍|机构|姓名|电话号码|街道地址|称谓|驾驶证号|社会保障号码|邮政编码",
    "es": "edad|número de tarjeta|fecha|sitio web|correo electrónico|ciudad|IBAN|dirección IP|nacionalidad|organización|nombre completo|teléfono|dirección postal|título|permiso de conducir|número de seguridad social|código postal",
    "fr": "âge|numéro de carte|date|site web|adresse courriel|ville|IBAN|adresse IP|nationalité|organisation|nom complet|numéro de téléphone|adresse|titre|permis de conduire|numéro de sécurité sociale|code postal",
    "de": "Alter|Kartennummer|Datum|Webseite|E-Mail-Adresse|Stadt|IBAN|IP-Adresse|Nationalität|Organisation|vollständiger Name|Telefonnummer|Straßenadresse|Titel|Führerscheinnummer|Sozialversicherungsnummer|Postleitzahl",
    "ja": "年齢|カード番号|日付|ウェブサイト|メールアドレス|市|IBAN|IPアドレス|国籍|組織|氏名|電話番号|住所|敬称|運転免許証番号|社会保障番号|郵便番号",
}
TEMPLATES = {
    "en": [
        "The recorded {field} is {value}.",
        "Please save this {field}: {value}.",
        "For the account, {field} = {value}.",
        "The form lists [{value}] under {field}.",
        "A support request mentions {value} as the {field}.",
        "The supplied detail, {value}, corresponds to {field}.",
    ],
    "zh": [
        "记录中的{field}是{value}。",
        "请保存这项{field}：{value}。",
        "账户资料：{field} = {value}。",
        "表格在{field}一栏填写了「{value}」。",
        "一份咨询请求提到，{field}为{value}。",
        "提供的信息{value}对应{field}。",
    ],
    "es": [
        "El dato de {field} es {value}.",
        "Guarde este dato de {field}: {value}.",
        "Para la cuenta, {field} = {value}.",
        "El formulario indica [{value}] en {field}.",
        "Una solicitud menciona {value} como {field}.",
        "El dato proporcionado, {value}, corresponde a {field}.",
    ],
    "fr": [
        "La valeur de {field} est {value}.",
        "Veuillez enregistrer {field} : {value}.",
        "Pour le compte, {field} = {value}.",
        "Le formulaire indique [{value}] dans {field}.",
        "Une demande mentionne {value} comme {field}.",
        "Le renseignement fourni, {value}, correspond à {field}.",
    ],
    "de": [
        "Der Eintrag für {field} ist {value}.",
        "Bitte speichern Sie {field}: {value}.",
        "Für das Konto gilt {field} = {value}.",
        "Das Formular enthält [{value}] unter {field}.",
        "Eine Anfrage nennt {value} als {field}.",
        "Die Angabe {value} gehört zum Feld {field}.",
    ],
    "ja": [
        "記録された{field}は{value}です。",
        "この{field}を保存してください：{value}。",
        "アカウント情報：{field} = {value}。",
        "フォームの{field}欄には「{value}」とあります。",
        "問い合わせには{field}として{value}が記載されています。",
        "提供された情報{value}は{field}に対応します。",
    ],
}
NEGATIVES = {
    "en": [
        "The service retries a failed request after a short pause.",
        "A dot ends the sentence; the at sign is a symbol.",
        "The application can process a document and return a result.",
    ],
    "zh": [
        "服务会在短暂等待后重试失败的请求。",
        "句号用于结束句子，符号本身不代表个人资料。",
        "应用能够处理文档并返回结果。",
    ],
    "es": [
        "El servicio vuelve a intentar la solicitud tras una pausa.",
        "Un punto termina la frase; la arroba es un símbolo.",
        "La aplicación procesa un documento y devuelve un resultado.",
    ],
    "fr": [
        "Le service réessaie la requête après une courte pause.",
        "Un point termine la phrase ; l'arobase est un symbole.",
        "L'application traite un document et retourne un résultat.",
    ],
    "de": [
        "Der Dienst wiederholt die Anfrage nach einer kurzen Pause.",
        "Ein Punkt beendet den Satz; das At-Zeichen ist ein Symbol.",
        "Die Anwendung verarbeitet ein Dokument und liefert ein Ergebnis.",
    ],
    "ja": [
        "サービスは短い待機の後にリクエストを再試行します。",
        "句点は文の終わりを示し、記号だけでは個人情報になりません。",
        "アプリケーションは文書を処理して結果を返します。",
    ],
}
FIRST_NAMES = [
    [
        "Maya",
        "Amelia",
        "Haruto",
        "Luca",
        "Priya",
        "Omar",
        "Sofia",
        "Daniel",
        "Amina",
        "Felix",
        "Hana",
        "Victor",
    ],
    ["Elena", "Diego", "Ingrid", "Jonas", "Leila", "Marco", "Nia", "Pavel"],
    ["Noah", "Clara", "Darius", "Esther", "Farah", "Hugo", "Iris", "Kenji"],
]
LAST_NAMES = [
    [
        "Collins",
        "Patel",
        "Sato",
        "Rossi",
        "Haddad",
        "Fischer",
        "Costa",
        "Nielsen",
        "Park",
        "Mensah",
        "Okafor",
        "Novak",
    ],
    ["Serrano", "Bergstrom", "Moreau", "Suzuki", "Abbas", "Lindholm"],
    ["Bennett", "Dubois", "Tanaka", "Ibrahim", "Nowak", "Silva"],
]
CITIES = [
    ["Riverton", "Oslo", "Nairobi", "Lisbon", "Brno", "Dakar", "Lima", "Kyoto"],
    ["Valencia", "Aarhus", "Bologna", "Accra", "Hobart", "Nagoya"],
    ["Sapporo", "Bergen", "Coimbra", "Cuenca", "Graz", "Wellington"],
]
NATIONALITIES = [
    ["Canadian", "Kenyan", "Brazilian", "Swedish", "Indian", "Korean"],
    ["Portuguese", "Mexican", "Nigerian", "Finnish"],
    ["Icelandic", "Peruvian", "Vietnamese", "Hungarian"],
]
TITLES = [
    ["Dr.", "Mr.", "Ms.", "Mrs."],
    ["Professor", "Captain"],
    ["Senator", "Reverend"],
]


LUHN_HALF_BASE = 5


def luhn_number(prefix):
    digits = [int(char) for char in prefix]
    checksum = sum(
        (
            (2 * value - 9 if value >= LUHN_HALF_BASE else 2 * value)
            if index % 2 == 0
            else value
        )
        for index, value in enumerate(digits)
    )
    return prefix + str((-checksum) % 10)


def entity_value(entity_type, split, index):
    split_index = ["train", "dev", "test"].index(split)
    number = (split_index + 1) * 100000 + index
    first = FIRST_NAMES[split_index][index % len(FIRST_NAMES[split_index])]
    last = LAST_NAMES[split_index][
        (index // len(FIRST_NAMES[split_index])) % len(LAST_NAMES[split_index])
    ]
    word = ["harbor", "summit", "meadow"][split_index]
    if entity_type == "EMAIL_ADDRESS":
        local = [
            f"{first.lower()}{number}",
            f"{first.lower()}.{last.lower()}{number}",
            f"{first.lower()}+case{number}",
            f"{first.lower()}_{number}",
            f"{last.lower()}-{number}",
            f"{first.lower()}.{last.lower()}+case{number}",
        ][index % 6]
        return f"{local}@{word}{index % 29}.example.org"
    if entity_type == "CREDIT_CARD":
        return luhn_number("4" + f"{number:014d}")
    if entity_type == "IBAN_CODE":
        account = f"{number:014d}"
        digits = "29142829" + account + "161100"
        return "GB" + f"{98 - int(digits) % 97:02d}" + "TEST" + account
    values = {
        "AGE": str(20 + split_index * 20 + index % 20),
        "DATE_TIME": f"{2001 + split_index * 9 + index % 7}-{index % 12 + 1:02d}-{index % 27 + 1:02d}",
        "DOMAIN_NAME": (
            f"https://{word}{number}.example.net/status"
            if index % 2
            else f"{word}{number}.example.net"
        ),
        "GPE": CITIES[split_index][index % len(CITIES[split_index])],
        "IP_ADDRESS": ["192.0.2.", "198.51.100.", "203.0.113."][split_index]
        + str(index % 253 + 1),
        "NRP": NATIONALITIES[split_index][index % len(NATIONALITIES[split_index])],
        "ORGANIZATION": f"{word.capitalize()} Research {number}",
        "PERSON": f"{first} {last}",
        "PHONE_NUMBER": f"+{[1, 44, 81][split_index]}-{200 + index % 700}-{number % 1000:03d}-{number % 10000:04d}",
        "STREET_ADDRESS": f"{number} {word.capitalize()} Road",
        "TITLE": TITLES[split_index][index % len(TITLES[split_index])],
        "US_DRIVER_LICENSE": f"{['C', 'D', 'E'][split_index]}{number:08d}",
        "US_SSN": f"{200 + split_index * 200 + index % 100:03d}-{index % 80 + 10:02d}-{number % 10000:04d}",
        "ZIP_CODE": str(10000 + split_index * 20000 + index % 19000),
    }
    return values[entity_type]


def annotated_record(text, entity_type, value, **metadata):
    start = text.index(value)
    return {
        "full_text": text,
        "spans": [
            {
                "entity_type": entity_type,
                "entity_value": value,
                "start_position": start,
                "end_position": start + len(value),
            }
        ],
        **metadata,
    }


def generated_split(split, per_type, seed):
    rng = random.Random(seed)
    records = []
    families = {"train": [0, 1, 2], "dev": [3], "test": [4, 5]}[split]
    for language, fields in FIELDS.items():
        field_names = fields.split("|")
        if len(TYPES) != len(field_names):
            raise ValueError("Localized field names must cover every entity type")
        names = dict(zip(TYPES, field_names))  # noqa: B905 - Python 3.9 data tooling
        for entity_type in TYPES:
            for index in range(per_type * (3 if entity_type == "EMAIL_ADDRESS" else 1)):
                family = families[index % len(families)]
                value = entity_value(entity_type, split, index)
                text = TEMPLATES[language][family].format(
                    field=names[entity_type], value=value
                )
                if value.endswith("."):
                    value_end = text.index(value) + len(value)
                    if text[value_end : value_end + 1] in (".", "。"):
                        text = text[:value_end] + text[value_end + 1 :]
                    elif (
                        text[value_end : value_end + 1]
                        and not text[value_end].isspace()
                    ):
                        text = text[:value_end] + " " + text[value_end:]
                identifier = f"synthetic-{split}-{language}-{entity_type}-{index}"
                records.append(
                    annotated_record(
                        text,
                        entity_type,
                        value,
                        id=identifier,
                        language=language,
                        template_family=f"field-{family}",
                        source_group=identifier,
                        provenance="original-synthetic-v1",
                        split=split,
                        kind="positive",
                    )
                )
        for index in range(per_type):
            # Different negative sentence families belong to different splits.
            family = ["train", "dev", "test"].index(split)
            text = "\n".join([NEGATIVES[language][family]] * (index + 1))
            # Repeated-context stress samples are explicit, and all variants
            # of one negative source remain in the same partition.
            identifier = f"negative-{split}-{language}-{index}"
            records.append(
                {
                    "full_text": text,
                    "spans": [],
                    "id": identifier,
                    "language": language,
                    "template_family": f"negative-{family}",
                    "source_group": f"negative-{split}-{language}",
                    "provenance": "original-synthetic-v1",
                    "split": split,
                    "kind": "negative-stress",
                }
            )
    rng.shuffle(records)
    # Low-cardinality values (titles, nationalities, cities) can otherwise
    # repeat the same template verbatim and inflate evaluation row counts.
    unique = {}
    for record in records:
        previous = unique.setdefault(record["full_text"], record)
        if previous["spans"] != record["spans"]:
            raise ValueError("Identical text has conflicting annotations")
    return list(unique.values())


def write_jsonl(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content)
    return {
        "file": path.name,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "rows": len(records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--presidio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-type", type=int, default=60)
    parser.add_argument("--dev-per-type", type=int, default=8)
    parser.add_argument("--test-per-type", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    label_to_id, _ = load_label_contract(args.config)
    if set(label_to_id) != {"O"} | {
        prefix + entity_type for entity_type in TYPES for prefix in ("B-", "I-")
    }:
        raise ValueError(
            "Repair workflow requires the existing 17-type / 35-label contract"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a frozen corpus")
    splits = {
        split: generated_split(split, count, args.seed + index)
        for index, (split, count) in enumerate(
            [
                ("train", args.train_per_type),
                ("dev", args.dev_per_type),
                ("test", args.test_per_type),
            ]
        )
    }
    assert_split_isolation(splits)
    replay = []
    seen = set()
    for index, sample in enumerate(json.loads(args.presidio.read_text())):
        validated_spans(sample, label_to_id)
        if sample["full_text"] in seen:
            continue
        seen.add(sample["full_text"])
        replay.append(
            {
                **sample,
                "id": f"presidio-{index}",
                "language": "en",
                "source_group": f"presidio-template-{sample['template_id']}",
                "provenance": "presidio-synth-v2",
                "split": "replay",
                "kind": "retention",
            }
        )
    # The original checkpoint may already have seen all Presidio rows. These
    # rows are a retention set, not a claimed unseen evaluation split.
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        write_jsonl(args.output / f"{split}.jsonl", records)
        for split, records in splits.items()
    ]
    files.append(write_jsonl(args.output / "replay.jsonl", replay))
    manifest = {
        "version": 1,
        "seed": args.seed,
        "kind": "synthetic-repair-with-separate-seen-retention",
        "label2id": label_to_id,
        "source_sha256": hashlib.sha256(args.presidio.read_bytes()).hexdigest(),
        "split_rule": "Synthetic template families, source groups, texts and complete entity values are disjoint. Presidio is separately labeled seen retention; test rows never select checkpoints.",
        "files": files,
        "counts": {
            split: dict(
                Counter(
                    span["entity_type"]
                    for record in records
                    for span in record["spans"]
                )
            )
            for split, records in splits.items()
        },
        "limitations": [
            "Synthetic contexts and entity values are not production-distribution evidence.",
            "Existing runtime diagnostics are excluded from this corpus.",
            "Presidio may overlap the baseline checkpoint's original training data.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"files": files, "label_count": len(label_to_id)}))


if __name__ == "__main__":
    main()
