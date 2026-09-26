"""Build a private, DEV-only multilingual decision pilot from authored facts.

The casebook and gold are private. This public generator contains the oracle,
four independently written rendering templates, and structural checks only.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from inference.run import digest

from multilingual.audit import script_of, sha256

VERSION = "decision2-multilingual-hard-dev/1"
LANGUAGES = ("en", "zh", "es", "ja")
OPS = (
    "choice_latest",
    "choice_minimax",
    "choice_two_proofs",
    "noul_all",
    "noul_chain",
    "noul_revoke",
    "score_weighted",
    "score_penalty",
    "score_order",
)
KIND = {op: op.split("_", 1)[0] for op in OPS}
CHOICE_IDS = ("A", "B", "C", "HOLD")

# These are original locale templates for structured facts, not machine
# translations of an upstream benchmark. Native-speaker review is still
# required: token checks cannot establish semantic equivalence.
WORDS = {
    "en": {
        "rule": "Rule",
        "records": "Records",
        "record": "Record",
        "question": "Decide from these records only.",
        "day": "day",
        "scope": "scope",
        "issued": "issued",
        "expires": "expires",
        "signed": "signed",
        "withdrawn": "withdrawn",
        "version": "version",
        "spend": "spend",
        "red": "red loss",
        "blue": "blue loss",
        "budget": "budget",
        "owner": "owner verified",
        "safety": "safety passed",
        "cleared": "cleared",
        "exempt": "exempt",
        "suspended": "suspended",
        "approved": "approved",
        "grant": "grant",
        "revoke": "revoke",
        "marks": "marks",
        "weights": "weights",
        "thresholds": "thresholds",
        "late": "late events",
        "open": "open exceptions",
        "credit": "credit",
        "stages": "stages",
        "true": "yes",
        "false": "no",
        "choice": "Choose the supported record or HOLD.",
        "noul": "Does the stated approval condition hold?",
        "score": "What is the level, from 0 through 3?",
        "hold": "HOLD: no eligible record",
        "level": "level",
    },
    "zh": {
        "rule": "规则",
        "records": "记录",
        "record": "记录",
        "question": "只根据这些记录判断。",
        "day": "日期",
        "scope": "范围",
        "issued": "签发",
        "expires": "有效至",
        "signed": "已签署",
        "withdrawn": "已撤回",
        "version": "版本",
        "spend": "费用",
        "red": "红色损失",
        "blue": "蓝色损失",
        "budget": "预算",
        "owner": "所有者已核验",
        "safety": "安全审查通过",
        "cleared": "已获准",
        "exempt": "获明确豁免",
        "suspended": "被暂停",
        "approved": "已批准",
        "grant": "授权",
        "revoke": "撤销",
        "marks": "得分标记",
        "weights": "权重",
        "thresholds": "门槛",
        "late": "逾期事件",
        "open": "未结例外",
        "credit": "抵扣",
        "stages": "阶段",
        "true": "是",
        "false": "否",
        "choice": "选择有依据的记录；都不符合则选 HOLD。",
        "noul": "所述批准条件是否成立？",
        "score": "等级是多少？范围为 0 到 3。",
        "hold": "HOLD：无合格记录",
        "level": "等级",
    },
    "es": {
        "rule": "Regla",
        "records": "Registros",
        "record": "Registro",
        "question": "Decida solo con estos registros.",
        "day": "día",
        "scope": "ámbito",
        "issued": "emitido",
        "expires": "válido hasta",
        "signed": "firmado",
        "withdrawn": "retirado",
        "version": "versión",
        "spend": "costo",
        "red": "pérdida roja",
        "blue": "pérdida azul",
        "budget": "presupuesto",
        "owner": "titular verificado",
        "safety": "seguridad aprobada",
        "cleared": "habilitado",
        "exempt": "exento expresamente",
        "suspended": "suspendido",
        "approved": "aprobado",
        "grant": "concesión",
        "revoke": "revocación",
        "marks": "marcas",
        "weights": "pesos",
        "thresholds": "umbrales",
        "late": "demoras",
        "open": "excepciones abiertas",
        "credit": "crédito",
        "stages": "etapas",
        "true": "sí",
        "false": "no",
        "choice": "Elija el registro respaldado, o HOLD si ninguno cumple.",
        "noul": "¿Se cumple la condición de aprobación indicada?",
        "score": "¿Cuál es el nivel, entre 0 y 3?",
        "hold": "HOLD: ningún registro cumple",
        "level": "nivel",
    },
    "ja": {
        "rule": "規則",
        "records": "記録",
        "record": "記録",
        "question": "これらの記録だけで判断してください。",
        "day": "日",
        "scope": "対象",
        "issued": "発行日",
        "expires": "有効期限",
        "signed": "署名済み",
        "withdrawn": "撤回済み",
        "version": "版",
        "spend": "費用",
        "red": "赤の損失",
        "blue": "青の損失",
        "budget": "予算",
        "owner": "所有者確認済み",
        "safety": "安全審査通過",
        "cleared": "許可済み",
        "exempt": "明示的に免除",
        "suspended": "停止中",
        "approved": "承認済み",
        "grant": "許可日",
        "revoke": "取消日",
        "marks": "得点",
        "weights": "重み",
        "thresholds": "しきい値",
        "late": "遅延件数",
        "open": "未解決の例外",
        "credit": "加点",
        "stages": "段階",
        "true": "はい",
        "false": "いいえ",
        "choice": "根拠のある記録を選び、該当しなければ HOLD を選んでください。",
        "noul": "指定された承認条件は成立しますか。",
        "score": "0 から 3 のどの段階ですか。",
        "hold": "HOLD：該当記録なし",
        "level": "段階",
    },
}

RULES = {
    "choice_latest": {
        "en": "Only a signed, not withdrawn, matching-scope bulletin issued by the decision day and unexpired on that day qualifies. Choose the qualifying bulletin with the latest issue day; otherwise HOLD.",
        "zh": "仅已签署、未撤回、范围相符，且在决策日当日或之前签发并于当日仍有效的公告合格。选签发日最晚的合格公告；否则选 HOLD。",
        "es": "Solo cumple un aviso firmado, no retirado, del mismo ámbito, emitido como máximo el día de decisión y aún vigente ese día. Elija el aviso válido más reciente; si no hay ninguno, HOLD.",
        "ja": "署名済みで撤回されず、対象が一致し、判断日までに発行されてその日も有効な通知だけが対象です。対象のうち発行日が最も新しいものを選び、なければ HOLD とします。",
    },
    "choice_minimax": {
        "en": "Exclude plans over budget. Among the rest choose the smallest of each plan's worse red/blue loss; break ties by lower spend, then alphabetic ID. If none fits, HOLD.",
        "zh": "费用超过预算的方案先排除。其余方案各取红、蓝两种情形中较大的损失，选择该值最小者；同值时先比费用低，再比字母顺序。无合格方案选 HOLD。",
        "es": "Descarte los planes que exceden el presupuesto. De los restantes, minimice la mayor pérdida entre rojo y azul; en empate, menor costo y luego ID alfabético. Si no queda ninguno, HOLD.",
        "ja": "予算超過の案を除きます。残りの各案について赤と青の損失の大きい方を比較し、最小の案を選びます。同点なら費用、さらにIDの辞書順で決めます。該当がなければ HOLD です。",
    },
    "choice_two_proofs": {
        "en": "A record qualifies only when both owner verification and safety approval are present and it is not withdrawn. Choose the qualifying record with the greatest version; otherwise HOLD.",
        "zh": "所有者核验和安全审查均通过且未撤回的记录才合格。选择版本号最高的合格记录；否则选 HOLD。",
        "es": "Un registro cumple solo si se ha verificado al titular, se ha aprobado la seguridad y no fue retirado. Elija la versión válida más alta; si no hay ninguna, HOLD.",
        "ja": "所有者の確認と安全審査の両方が済み、撤回されていない記録だけが対象です。対象のうち版番号が最大のものを選び、なければ HOLD とします。",
    },
    "noul_all": {
        "en": "Approve only if every listed person is cleared or explicitly exempt. A suspension overrides either status. An absent clearance or exemption is not evidence of approval.",
        "zh": "列出的每个人都已获准或获明确豁免时才批准。暂停状态优先于前两者。缺少准许或豁免记录不能算作批准。",
        "es": "Apruebe solo si cada persona está habilitada o exenta expresamente. Una suspensión prevalece sobre ambos estados. La ausencia de constancia no equivale a aprobación.",
        "ja": "記載された全員が許可済みか明示的に免除されている場合だけ承認します。停止中ならどちらの状態よりも優先して不承認です。記録がない場合は承認の根拠になりません。",
    },
    "noul_chain": {
        "en": "The A→B→C chain is authorized only if both directed links are approved, not withdrawn, and unexpired on the decision day. A reverse-direction link does not count.",
        "zh": "A→B→C 链仅在两条有方向的连接都已批准、未撤回且在决策日仍有效时才获授权。反向连接不能替代。",
        "es": "La cadena A→B→C se autoriza solo si los dos enlaces dirigidos están aprobados, no retirados y vigentes el día de decisión. El enlace inverso no sirve.",
        "ja": "A→B→C の連鎖は、両方の有向接続が承認済みで撤回されず、判断日に有効な場合だけ認めます。逆向きの接続は代わりになりません。",
    },
    "noul_revoke": {
        "en": "Approval holds on the decision day only after a grant, through its expiry day, and only if no revocation took effect on or before that day. A later revocation is irrelevant.",
        "zh": "仅在授权已经生效、决策日不晚于到期日，并且没有在决策日或更早生效的撤销时，当日的批准才成立。之后的撤销不影响当日判断。",
        "es": "La aprobación vale el día de decisión solo desde la concesión hasta su vencimiento y si no hay revocación efectiva ese día o antes. Una revocación posterior no cuenta.",
        "ja": "判断日の承認が成立するのは、許可日以降かつ有効期限内で、その日以前に効力が生じた取消しがない場合だけです。後日の取消しは当日の判断に影響しません。",
    },
    "score_weighted": {
        "en": "Multiply each binary mark by its paired weight and sum. Level equals the number of thresholds 2, 4, 6 that the sum reaches, including equality.",
        "zh": "各二值得分标记与其对应权重相乘后求和。总和达到门槛 2、4、6 中的几个，等级就是几；相等也算达到。",
        "es": "Multiplique cada marca binaria por su peso emparejado y sume. El nivel es el número de umbrales 2, 4 y 6 alcanzados, incluida la igualdad.",
        "ja": "各二値の得点に対応する重みを掛けて合計します。合計がしきい値 2、4、6 のそれぞれに達しているかを調べ、達したしきい値の数を段階とします。等しい場合も達成です。",
    },
    "score_penalty": {
        "en": "Begin at level 3. Subtract one per late event and two per open exception, then add the mitigation credit. Clamp the result to 0..3.",
        "zh": "从等级 3 开始。每个逾期事件减 1，每个未结例外减 2，最后加抵扣。结果限制在 0 到 3 之间。",
        "es": "Empiece en nivel 3. Descuente uno por cada demora y dos por cada excepción abierta; después sume el crédito de mitigación. Mantenga el resultado dentro de 0..3.",
        "ja": "段階 3 から始め、遅延1件につき1、未解決の例外1件につき2を引き、最後に軽減の加点を足します。結果は 0 から 3 に収めます。",
    },
    "score_order": {
        "en": "Stages 1, 2, 3 must pass in order. Level is the number of consecutive passed stages starting at 1; a later pass cannot repair an earlier failure.",
        "zh": "阶段 1、2、3 必须依次通过。等级是从阶段 1 开始连续通过的阶段数；后续通过不能弥补前面的未通过。",
        "es": "Las etapas 1, 2 y 3 deben superarse en orden. El nivel es la cantidad de etapas consecutivas aprobadas desde la 1; una etapa posterior no repara un fallo anterior.",
        "ja": "段階 1、2、3 は順に通過する必要があります。段階 1 から連続して通過した数が評価値です。後の通過で前の不合格を補うことはできません。",
    },
}


def oracle(case: dict) -> str | bool | int:
    """Compute the key from structured facts; never trust a stored gold label."""
    op, facts = case["operation"], case["facts"]
    if op == "choice_latest":
        eligible = [
            row
            for row in facts["records"]
            if row["scope"] == facts["scope"]
            and row["signed"]
            and not row["withdrawn"]
            and row["issued"] <= facts["day"] <= row["expires"]
        ]
        return (
            max(eligible, key=lambda row: (row["issued"], -ord(row["id"][0])))["id"]
            if eligible
            else "HOLD"
        )
    if op == "choice_minimax":
        eligible = [row for row in facts["records"] if row["spend"] <= facts["budget"]]
        return (
            min(
                eligible,
                key=lambda row: (max(row["red"], row["blue"]), row["spend"], row["id"]),
            )["id"]
            if eligible
            else "HOLD"
        )
    if op == "choice_two_proofs":
        eligible = [
            row
            for row in facts["records"]
            if row["owner"] and row["safety"] and not row["withdrawn"]
        ]
        return (
            max(eligible, key=lambda row: (row["version"], -ord(row["id"][0])))["id"]
            if eligible
            else "HOLD"
        )
    if op == "noul_all":
        return all(
            (row["cleared"] or row["exempt"]) and not row["suspended"]
            for row in facts["people"]
        )
    if op == "noul_chain":
        required = {("A", "B"), ("B", "C")}
        valid = {
            (row["from"], row["to"])
            for row in facts["links"]
            if row["approved"]
            and not row["withdrawn"]
            and row["expires"] >= facts["day"]
        }
        return required <= valid
    if op == "noul_revoke":
        return facts["grant"] <= facts["day"] <= facts["expires"] and all(
            day > facts["day"] for day in facts["revocations"]
        )
    if op == "score_weighted":
        total = sum(
            mark * weight
            for mark, weight in zip(facts["marks"], facts["weights"], strict=True)
        )
        return sum(total >= threshold for threshold in (2, 4, 6))
    if op == "score_penalty":
        return max(0, min(3, 3 - facts["late"] - 2 * facts["open"] + facts["credit"]))
    if op == "score_order":
        level = 0
        for passed in facts["stages"]:
            if not passed:
                break
            level += 1
        return level
    raise ValueError(f"Unknown operation: {op}")


def _flag(value: bool, words: dict) -> str:
    return words["true"] if value else words["false"]


def render(case: dict, language: str) -> dict:
    if language not in LANGUAGES:
        raise ValueError("Unknown language")
    op, facts = case["operation"], case["facts"]
    words = WORDS[language]
    head = f"{words['rule']}: {RULES[op][language]}\n"
    lines = []
    if op == "choice_latest":
        lines.append(
            f"{words['day']}={facts['day']}; {words['scope']}={facts['scope']}"
        )
        for row in facts["records"]:
            lines.append(
                f"{row['id']}: {words['scope']}={row['scope']}; "
                f"{words['issued']}={row['issued']}; {words['expires']}={row['expires']}; "
                f"{words['signed']}={_flag(row['signed'], words)}; "
                f"{words['withdrawn']}={_flag(row['withdrawn'], words)}"
            )
    elif op == "choice_minimax":
        lines.append(f"{words['budget']}={facts['budget']}")
        for row in facts["records"]:
            lines.append(
                f"{row['id']}: {words['spend']}={row['spend']}; "
                f"{words['red']}={row['red']}; {words['blue']}={row['blue']}"
            )
    elif op == "choice_two_proofs":
        for row in facts["records"]:
            lines.append(
                f"{row['id']}: {words['version']}={row['version']}; "
                f"{words['owner']}={_flag(row['owner'], words)}; "
                f"{words['safety']}={_flag(row['safety'], words)}; "
                f"{words['withdrawn']}={_flag(row['withdrawn'], words)}"
            )
    elif op == "noul_all":
        for row in facts["people"]:
            lines.append(
                f"{row['id']}: {words['cleared']}={_flag(row['cleared'], words)}; "
                f"{words['exempt']}={_flag(row['exempt'], words)}; "
                f"{words['suspended']}={_flag(row['suspended'], words)}"
            )
    elif op == "noul_chain":
        lines.append(f"{words['day']}={facts['day']}")
        for row in facts["links"]:
            lines.append(
                f"{row['from']}→{row['to']}: {words['approved']}={_flag(row['approved'], words)}; "
                f"{words['withdrawn']}={_flag(row['withdrawn'], words)}; "
                f"{words['expires']}={row['expires']}"
            )
    elif op == "noul_revoke":
        lines.append(
            f"{words['day']}={facts['day']}; {words['grant']}={facts['grant']}; "
            f"{words['expires']}={facts['expires']}; "
            f"{words['revoke']}={','.join(map(str, facts['revocations'])) or 'none'}"
        )
    elif op == "score_weighted":
        lines.append(
            f"{words['marks']}={','.join(map(str, facts['marks']))}; "
            f"{words['weights']}={','.join(map(str, facts['weights']))}; "
            f"{words['thresholds']}=2,4,6"
        )
    elif op == "score_penalty":
        lines.append(
            f"{words['late']}={facts['late']}; {words['open']}={facts['open']}; "
            f"{words['credit']}={facts['credit']}"
        )
    elif op == "score_order":
        lines.append(
            f"{words['stages']}="
            + ", ".join(
                f"{i}:{_flag(passed, words)}"
                for i, passed in enumerate(facts["stages"], 1)
            )
        )
    else:
        raise ValueError(op)
    state = head + f"{words['records']}:\n" + "\n".join(lines)
    question = {
        "type": KIND[op],
        "instructions": words[KIND[op]] + " " + words["question"],
    }
    if KIND[op] == "choice":
        options = case["option_order"]
        question["criteria"] = {
            chr(97 + i): (
                words["hold"] if item == "HOLD" else f"{words['record']} {item}"
            )
            for i, item in enumerate(options)
        }
    elif KIND[op] == "noul":
        question["criteria"] = {"false": words["false"], "true": words["true"]}
    else:
        question["criteria"] = [f"{words['level']} {i}" for i in range(4)]
    return {
        "id": f"mlhard-{case['id']}-{language}",
        "state": state,
        "questions": {"decision": question},
    }


def load_cases(path: Path) -> list[dict]:
    if "final" in str(path).casefold():
        raise ValueError("Never load a sealed final source")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 18:
        raise ValueError("Expected exactly 18 original base cases")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != 18 or any(
        not re.fullmatch(r"[a-z][a-z0-9-]+", name) for name in ids
    ):
        raise ValueError("Duplicate or malformed case ID")
    operations = Counter(case["operation"] for case in cases)
    if set(operations) != set(OPS) or set(operations.values()) != {2}:
        raise ValueError("Require two original cases for every operation")
    for case in cases:
        op = case["operation"]
        if KIND[op] == "choice" and sorted(case["option_order"]) != sorted(CHOICE_IDS):
            raise ValueError("Choice must expose each semantic option once")
        if "gold" in case or "answer" in case:
            raise ValueError("Do not store a hand-written gold label")
        result = oracle(case)
        if KIND[op] == "choice" and result not in CHOICE_IDS:
            raise ValueError("Choice oracle result is not available")
        if KIND[op] == "noul" and type(result) is not bool:
            raise ValueError("Noul oracle result must be boolean")
        if KIND[op] == "score" and (type(result) is not int or not 0 <= result <= 3):
            raise ValueError("Score oracle result out of bounds")
    return cases


def make_rows(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    prompts, targets = [], []
    for case in cases:
        semantic = oracle(case)
        for language in LANGUAGES:
            prompt = render(case, language)
            target = {
                "id": prompt["id"],
                "base_id": case["id"],
                "language": language,
                "task_type": KIND[case["operation"]],
                "operation": case["operation"],
                "semantic_gold": semantic,
                "source_input_sha256": digest(
                    {"state": prompt["state"], "questions": prompt["questions"]}
                ),
            }
            if target["task_type"] == "choice":
                target["gold"] = chr(97 + case["option_order"].index(semantic))
                target["semantic_by_label"] = {
                    chr(97 + i): value for i, value in enumerate(case["option_order"])
                }
            elif target["task_type"] == "noul":
                target["gold"] = semantic
            else:
                target["gold"] = semantic
            if any(key in prompt for key in ("gold", "answer", "facts", "operation")):
                raise ValueError("Prompt leaked private oracle data")
            prompts.append(prompt)
            targets.append(target)
    if len(prompts) != 72 or len({row["id"] for row in prompts}) != 72:
        raise ValueError("Expected 18 bases × 4 languages")
    for language in ("zh", "ja"):
        language_prompts = [
            row for row in prompts if row["id"].endswith("-" + language)
        ]
        expected_script = "han" if language == "zh" else "kana"
        if any(
            sum(script_of(ch) == expected_script for ch in row["state"]) < 8
            for row in language_prompts
        ):
            raise ValueError(f"{language}: local script missing")
    return prompts, targets


def _normalize(value: object) -> str:
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        if not isinstance(value, str)
        else value
    )
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def screen_overlap(prompts: list[dict], references: list[Path]) -> dict:
    """Conservative visible DEV overlap screen; a semantic audit remains needed."""
    candidate = [_normalize(row["state"]) for row in prompts]
    reference_texts = []
    reference_hashes = []
    for path in references:
        if "final" in str(path).casefold():
            raise ValueError("Never open a sealed final source")
        reference_hashes.append(sha256(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            state = row.get("state")
            if state is not None:
                reference_texts.append(_normalize(state))
    exact = set(candidate) & set(reference_texts)
    near = 0
    for left in candidate:
        for right in reference_texts:
            if left == right:
                continue
            if abs(len(left) - len(right)) > 0.10 * max(len(left), len(right)):
                continue
            if SequenceMatcher(None, left, right).ratio() >= 0.94:
                near += 1
    return {
        "reference_file_sha256": reference_hashes,
        "reference_rows_with_state": len(reference_texts),
        "exact_matches": len(exact),
        "near_matches": near,
        "method": "NFKC/casefold/whitespace + SequenceMatcher>=0.94 for comparable lengths",
        "limitation": "not a semantic or sealed-final decontamination guarantee",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o600)


def build(casebook: Path, output: Path, references: list[Path]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    cases = load_cases(casebook)
    prompts, targets = make_rows(cases)
    overlap = screen_overlap(prompts, references)
    if overlap["exact_matches"] or overlap["near_matches"]:
        raise ValueError("Visible reference overlap; quarantine whole base group")
    output.mkdir(mode=0o700, parents=True)
    _write_jsonl(output / "prompts.jsonl", prompts)
    _write_jsonl(output / "targets.private.jsonl", targets)
    review = [
        {
            "id": row["id"],
            "base_id": target["base_id"],
            "language": target["language"],
            "task_type": target["task_type"],
            "state": row["state"],
            "question": row["questions"]["decision"],
        }
        for row, target in zip(prompts, targets, strict=True)
    ]
    _write_jsonl(output / "review.gold-free.jsonl", review)
    counts = {
        "prompts": len(prompts),
        "independent_base_cases": len(cases),
        "languages": list(LANGUAGES),
        "by_language": dict(Counter(target["language"] for target in targets)),
        "by_type": dict(Counter(target["task_type"] for target in targets)),
        "by_operation": dict(Counter(target["operation"] for target in targets)),
        "gold_distribution_private": dict(
            Counter(
                f"{target['task_type']}:{target['semantic_gold']}" for target in targets
            )
        ),
    }
    public_manifest = {
        "schema_version": VERSION,
        "scope": "private development pilot; no sealed final or training",
        "generator_and_native_adapter_sha256": sha256(Path(__file__)),
        "scorer_sha256": sha256(Path(__file__).with_name("hard_score.py")),
        "native_interpreter_sha256": sha256(Path(__file__).with_name("score.py")),
        "inference_eligible": False,
        "training_approved": False,
        "publication_eligible": False,
        "independent_bilingual_review": "pending",
        "independence_unit": "base_id; language variants are paired",
        "files_sha256": {
            "prompts.jsonl": sha256(output / "prompts.jsonl"),
            "review.gold-free.jsonl": sha256(output / "review.gold-free.jsonl"),
        },
        "private_casebook_sha256": sha256(casebook),
        "private_targets_sha256": sha256(output / "targets.private.jsonl"),
        "overlap_screen": overlap,
        "counts": {
            key: value
            for key, value in counts.items()
            if key != "gold_distribution_private"
        },
        "quality_limits": [
            "Original cases and four authored locale templates have no independent bilingual review",
            "Mechanical code/number/script checks do not prove parallel meaning",
            "Template-rendered tables may admit shortcut strategies",
            "Visible-reference overlap does not cover sealed FINAL",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(public_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "private-counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "private-counts.json").chmod(0o600)
    return {
        "manifest": public_manifest,
        "manifest_sha256": sha256(output / "manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.casebook, args.output, args.reference), sort_keys=True))


if __name__ == "__main__":
    main()
