"""Freeze a new private, gold-blind multilingual hard DEV candidate.

The structured casebook is supplied privately. This module contains only
locale rendering, mechanical QA, and artifact commitments. It never runs a
model or reads sealed FINAL labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from inference.run import digest

from multilingual import hard_pilot as base
from multilingual.audit import script_of, sha256

VERSION = "decision2-multilingual-hard-dev/7"
LANGUAGES = base.LANGUAGES
KIND = base.KIND

WORDS = {
    language: {
        **base.WORDS[language],
        **({"credit": "减罚后的加分"} if language == "zh" else {}),
    }
    for language in LANGUAGES
}

# Original locale policies. They intentionally spell out inclusive day
# boundaries. Editorial and qualified bilingual review are still required.
POLICY = {
    "choice_latest": {
        "en": "An issue on or before the decision day counts, and an expiry on or after that day counts; both boundary days are included. A bulletin also needs the requested scope, signature and no withdrawal. Take the usable bulletin issued most recently (ID order breaks a tie), or HOLD if none qualifies.",
        "zh": "签发日不晚于决策日、到期日不早于决策日时，两端日期当天都计入有效期。公告还必须范围相符、已签署且未撤回。选择合格公告中签发日最晚的；同日按编号字母顺序；均不合格则选 HOLD。",
        "es": "Se incluyen ambos días límite: emisión igual o anterior al día de decisión y vencimiento igual o posterior. El aviso también necesita el ámbito pedido, firma y ausencia de retirada. Entre los aptos, elija el de emisión más reciente (desempate por ID); si no hay ninguno, HOLD.",
        "ja": "発行日は判断日以前、期限日は判断日以後なら有効で、両端の日を含みます。通知には対象の一致、署名、撤回されていないことも必要です。条件を満たす中で発行日が最新のものを選び、同日ならID順、なければ HOLD とします。",
    },
    "choice_minimax": {
        "en": "First remove any plan whose spend is above the budget. For each remaining plan compare its two losses by taking the larger one. Choose the smallest such worst loss; break ties by spend and then ID. With no affordable plan choose HOLD.",
        "zh": "先剔除费用高于预算的方案。对每个剩余方案取两种损失中的较大值，选择该最坏损失最小的方案；同值先按费用、再按编号排序。没有预算内方案则选 HOLD。",
        "es": "Primero quite los planes cuyo costo supera el presupuesto. De cada plan restante tome la mayor de sus dos pérdidas y elija el menor de esos peores resultados; desempate por costo e ID. Si ninguno es asequible, HOLD.",
        "ja": "費用が予算を超える案を先に除きます。残りの案ごとに二つの損失の大きい方を取り、その最悪損失が最小の案を選びます。同点なら費用、次にID順です。予算内の案がなければ HOLD です。",
    },
    "choice_two_proofs": {
        "en": "A record is usable only with both owner and safety proofs and without withdrawal. Among usable records select the highest version (ID order breaks a tie). If all lack required evidence, HOLD.",
        "zh": "记录须同时具备所有者核验与安全审查两项证据，而且不得已撤回。合格记录中选版本号最高者；同版本按编号字母顺序。没有合格记录则选 HOLD。",
        "es": "Un registro sirve solo con ambas pruebas, titular y seguridad, y sin retirada. Entre los aptos seleccione la versión más alta (desempate por ID). Si ninguno aporta todas las pruebas, HOLD.",
        "ja": "記録が有効なのは所有者確認と安全審査の両方があり、撤回されていない場合だけです。有効な中で版番号が最大のものを選び、同点ならID順です。該当がなければ HOLD です。",
    },
    "noul_all": {
        "en": "Every listed person must have either clearance or an explicit exemption. A suspension defeats either status. Missing evidence cannot be treated as approval.",
        "zh": "列出的每个人都必须有准许记录或明确豁免；一旦暂停，前述任一状态均不生效。缺少证据不能推定批准。",
        "es": "Cada persona listada necesita autorización o exención explícita. Una suspensión invalida cualquiera de las dos. La falta de prueba no permite presumir aprobación.",
        "ja": "記載された全員に許可または明示的な免除が必要です。停止中ならそのどちらよりも優先して無効になります。証拠の欠如を承認とみなしてはいけません。",
    },
    "noul_chain": {
        "en": "The directed A→B and B→C links must both be approved and unwithdrawn. An expiry equal to the decision day is still valid; an earlier expiry is not. A reverse link cannot replace either required direction.",
        "zh": "A→B 和 B→C 两条有向连接必须都已批准且未撤回。到期日等于决策日仍有效；早于决策日则无效。反向连接不能替代所需方向。",
        "es": "Ambos enlaces dirigidos, A→B y B→C, deben estar aprobados y no retirados. Un vencimiento igual al día de decisión sigue vigente; uno anterior no. El enlace inverso no sustituye al requerido.",
        "ja": "有向の A→B と B→C の両方が承認済みで撤回されていないことが必要です。期限日が判断日と同じなら有効で、前日以前なら無効です。逆向きの接続では代用できません。",
    },
    "noul_revoke": {
        "en": "The grant starts on its stated day and remains valid through the expiry day; both days are included. A revocation takes effect on its stated day, so one dated on or before the decision day defeats approval. A revocation the next day does not.",
        "zh": "授权从所列授权日当天开始，到到期日当天结束；两端日期均计入。撤销从所列日期当天生效，因此决策日当天或之前的撤销使批准失效；次日撤销不影响当日。",
        "es": "La concesión comienza en su fecha y sigue válida incluido el día de vencimiento; se incluyen ambos extremos. Una revocación surte efecto en su propia fecha: si es el día de decisión o antes, anula la aprobación; si es al día siguiente, no.",
        "ja": "許可は記載の許可日当日から期限日当日まで有効で、両端を含みます。取消しは記載日当日に効力を生じるため、判断日以前または同日の取消しは承認を無効にし、翌日の取消しは当日に影響しません。",
    },
    "score_weighted": {
        "en": "Pair each binary mark with the weight in the same position and add their products. Count how many of 2, 4 and 6 the total reaches; equality counts. That count is the level.",
        "zh": "按位置配对二值标记与权重，将各乘积相加。统计总和达到 2、4、6 三个门槛中的几个；等于门槛也算达到。这个数量就是等级。",
        "es": "Empareje cada marca binaria con el peso de la misma posición y sume los productos. Cuente cuántos umbrales 2, 4 y 6 alcanza el total; la igualdad cuenta. Ese número es el nivel.",
        "ja": "各二値マークと同じ位置の重みを掛け合わせて合計します。合計が 2、4、6 のしきい値のいくつに達するか数えます。等しい場合も達成とし、その数を段階とします。",
    },
    "score_penalty": {
        "en": "Start at 3, subtract one per late event and two per open exception, then ADD the mitigation credit. The credit increases the result; it is not another penalty. Bound the final level to 0..3.",
        "zh": "从 3 开始，每个逾期事件减 1，每个未结例外减 2；然后把减轻措施的加分加上。加分会提高结果，并不是再扣分。最终等级限制在 0 到 3。",
        "es": "Comience en 3, descuente uno por cada demora y dos por cada excepción abierta; después SUME la bonificación de mitigación. Esta bonificación aumenta el resultado, no es otra penalización. Mantenga el nivel final dentro de 0..3.",
        "ja": "3 から始め、遅延1件ごとに1、未解決の例外1件ごとに2を引き、その後で軽減措置の加点を足します。加点は値を増やすもので、追加の減点ではありません。最終段階は 0～3 に収めます。",
    },
    "score_order": {
        "en": "Check stages 1, 2 and 3 in that order. The level is how many stages pass consecutively from stage 1. A later pass never fills an earlier gap.",
        "zh": "按 1、2、3 的顺序检查阶段。等级等于从阶段 1 起连续通过的阶段数；后面通过的阶段不能补上前面的缺口。",
        "es": "Revise las etapas 1, 2 y 3 en ese orden. El nivel es el número de etapas superadas sin interrupción desde la primera. Un aprobado posterior no cubre un fallo previo.",
        "ja": "段階 1、2、3 の順に確認します。段階 1 から途切れずに通過した数が評価値です。後の通過で前の欠落を埋めることはできません。",
    },
}


def render(case: dict, language: str) -> dict:
    """Write a localized policy and evidence sheet in the native typed shape."""
    if language not in LANGUAGES:
        raise ValueError("Unknown locale")
    op, facts = case["operation"], case["facts"]
    w = WORDS[language]
    yes = lambda value: base._flag(value, w)
    evidence: list[str] = []
    if op == "choice_latest":
        evidence.append(f"{w['day']}: {facts['day']} | {w['scope']}: {facts['scope']}")
        evidence.extend(
            f"[{r['id']}] {w['scope']} {r['scope']} · {w['issued']} {r['issued']} · "
            f"{w['expires']} {r['expires']} · {w['signed']} {yes(r['signed'])} · "
            f"{w['withdrawn']} {yes(r['withdrawn'])}"
            for r in facts["records"]
        )
    elif op == "choice_minimax":
        evidence.append(f"{w['budget']}: {facts['budget']}")
        evidence.extend(
            f"[{r['id']}] ({w['spend']} {r['spend']}; {w['red']} {r['red']}; "
            f"{w['blue']} {r['blue']})"
            for r in facts["records"]
        )
    elif op == "choice_two_proofs":
        evidence.extend(
            f"[{r['id']}] {w['version']} {r['version']} / {w['owner']} {yes(r['owner'])} / "
            f"{w['safety']} {yes(r['safety'])} / {w['withdrawn']} {yes(r['withdrawn'])}"
            for r in facts["records"]
        )
    elif op == "noul_all":
        evidence.extend(
            f"[{r['id']}] {w['cleared']} {yes(r['cleared'])}; "
            f"{w['exempt']} {yes(r['exempt'])}; {w['suspended']} {yes(r['suspended'])}"
            for r in facts["people"]
        )
    elif op == "noul_chain":
        evidence.append(f"{w['day']}: {facts['day']}")
        evidence.extend(
            f"[{r['from']}→{r['to']}] {w['approved']} {yes(r['approved'])}; "
            f"{w['withdrawn']} {yes(r['withdrawn'])}; {w['expires']} {r['expires']}"
            for r in facts["links"]
        )
    elif op == "noul_revoke":
        evidence = [
            f"{w['day']}: {facts['day']}",
            f"{w['grant']}: {facts['grant']} → {w['expires']}: {facts['expires']}",
            f"{w['revoke']}: {', '.join(map(str, facts['revocations'])) or '—'}",
        ]
    elif op == "score_weighted":
        pairs = " + ".join(
            f"{mark}×{weight}"
            for mark, weight in zip(facts["marks"], facts["weights"], strict=True)
        )
        evidence = [
            f"{w['marks']} × {w['weights']}: {pairs}",
            f"{w['thresholds']}: 2 / 4 / 6",
        ]
    elif op == "score_penalty":
        evidence = [
            f"{w['late']}: {facts['late']}",
            f"{w['open']}: {facts['open']}",
            f"{w['credit']}: +{facts['credit']}",
        ]
    elif op == "score_order":
        evidence = [
            f"{w['stages']} {i}: {yes(passed)}"
            for i, passed in enumerate(facts["stages"], 1)
        ]
    else:
        raise ValueError(op)
    state = f"{w['rule']}: {POLICY[op][language]}\n{w['records']}:\n" + "\n".join(
        evidence
    )
    question = {
        "type": KIND[op],
        "instructions": w[KIND[op]] + " " + w["question"],
    }
    if KIND[op] == "choice":
        question["criteria"] = {
            chr(97 + i): w["hold"] if item == "HOLD" else f"{w['record']} {item}"
            for i, item in enumerate(case["option_order"])
        }
    elif KIND[op] == "noul":
        question["criteria"] = {"false": w["false"], "true": w["true"]}
    else:
        question["criteria"] = [f"{w['level']} {i}" for i in range(4)]
    return {
        "id": f"mlhard-v7-{case['id']}-{language}",
        "state": state,
        "questions": {"decision": question},
    }


def casebook_qa(cases: list[dict]) -> dict:
    """Apply prospective r7 structural, balance and shallow-heuristic gates."""
    if len(cases) != 18 or any(not c["id"].startswith("v7-") for c in cases):
        raise ValueError("r7 needs 18 freshly named bases")
    choices = [c for c in cases if KIND[c["operation"]] == "choice"]
    scores = [c for c in cases if KIND[c["operation"]] == "score"]
    semantic = Counter(base.oracle(c) for c in choices)
    native = Counter(chr(97 + c["option_order"].index(base.oracle(c))) for c in choices)
    if not 1 <= semantic["HOLD"] <= 2:
        raise ValueError("Require one or two genuine HOLD bases")
    if any(semantic[item] > 2 for item in "ABC") or max(native.values()) > 2:
        raise ValueError("Choice semantic/native positions are concentrated")
    levels = Counter(base.oracle(c) for c in scores)
    if set(levels) != set(range(4)) or max(levels.values()) > 2:
        raise ValueError("Score level spread is insufficient")
    weighted = [c for c in scores if c["operation"] == "score_weighted"]
    if (
        len({sum(c["facts"]["marks"]) for c in weighted}) != 1
        or len({base.oracle(c) for c in weighted}) != 2
    ):
        raise ValueError("Weighted pair permits a mark-count shortcut")
    ordered = [c for c in scores if c["operation"] == "score_order"]
    if (
        len({sum(c["facts"]["stages"]) for c in ordered}) != 1
        or len({base.oracle(c) for c in ordered}) != 2
    ):
        raise ValueError("Ordered pair permits a pass-count shortcut")
    credits = sorted(
        c["facts"]["credit"] for c in scores if c["operation"] == "score_penalty"
    )
    if credits[0] != 0 or credits[1] <= 0:
        raise ValueError("Need zero and positive mitigation additions")
    if not any(
        r["expires"] == c["facts"]["day"]
        for c in cases
        if c["operation"] in {"choice_latest", "noul_chain"}
        for r in c["facts"].get("records", c["facts"].get("links", []))
    ):
        raise ValueError("Missing same-day expiry boundary")
    if not any(
        c["facts"]["day"] + 1 in c["facts"]["revocations"]
        for c in cases
        if c["operation"] == "noul_revoke"
    ):
        raise ValueError("Missing next-day revocation boundary")
    for case in choices:
        rows = case["facts"]["records"]
        if {row["id"] for row in rows} != set("ABC") or len(rows) != 3:
            raise ValueError("Choice needs one fact record per A/B/C")
        if case["operation"] == "choice_latest":
            guess = max(rows, key=lambda row: row["issued"])["id"]
            if guess == base.oracle(case):
                raise ValueError("Newest-record shortcut wins a latest case")
    for case in scores:
        if case["operation"] != "score_weighted":
            continue
        marks = case["facts"]["marks"]
        for i in range(len(marks)):
            altered = {**case, "facts": {**case["facts"], "marks": marks.copy()}}
            altered["facts"]["marks"][i] = 1 - marks[i]
            if base.oracle(altered) == base.oracle(case):
                raise ValueError("Weighted mark has no outcome effect")
    # A rule-name/operation-only leave-one-out classifier predicts the other
    # case in each operation. Every pair must have distinct answers.
    for operation in base.OPS:
        pair = [c for c in cases if c["operation"] == operation]
        if len({base.oracle(c) for c in pair}) != 2:
            raise ValueError(f"Operation-only shortcut survives: {operation}")
    pivotality = evidence_pivotality(cases)
    return {
        "choice_hold_bases": semantic["HOLD"],
        "choice_fixed_semantic_ceiling": max(semantic.values()),
        "choice_fixed_native_ceiling": max(native.values()),
        "choice_never_hold_ceiling": 6 - semantic["HOLD"],
        "operation_only_leave_one_out_correct": 0,
        "score_fixed_level_ceiling": max(levels.values()),
        "weighted_equal_mark_count_distinct_levels": True,
        "order_equal_pass_count_distinct_levels": True,
        "evidence_pivotality": pivotality,
        "limitation": "Synthetic heuristics are not a learned keyword classifier; editorial counterfactual review remains pending",
    }


def evidence_pivotality(cases: list[dict]) -> dict:
    """Check direct Choice relevance and conditional conjunctive evidence.

    Conditional checks establish necessity in a repaired positive witness;
    they do not turn a noncausal row in a failed conjunction into a direct
    cause of that particular negative answer.
    """
    choice_records = 0
    roster_records = 0
    chain_links = 0
    weighted_marks = 0
    penalty_fields = 0
    order_stages = 0
    revocation_dates = 0
    for case in cases:
        op, facts = case["operation"], case["facts"]
        original = base.oracle(case)
        if op.startswith("choice_"):
            for index, record in enumerate(facts["records"]):
                changes: list[tuple[str, object]] = []
                if op == "choice_latest":
                    changes = [
                        ("scope", facts["scope"]),
                        ("scope", "_other_scope_"),
                        ("issued", facts["day"]),
                        ("issued", facts["day"] - 1),
                        ("expires", facts["day"] - 1),
                        ("expires", facts["day"]),
                        ("signed", not record["signed"]),
                        ("withdrawn", not record["withdrawn"]),
                    ]
                elif op == "choice_minimax":
                    changes = [
                        ("spend", facts["budget"]),
                        ("spend", facts["budget"] + 1),
                        ("red", 0),
                        ("red", 10),
                        ("blue", 0),
                        ("blue", 10),
                    ]
                else:
                    changes = [
                        ("owner", not record["owner"]),
                        ("safety", not record["safety"]),
                        ("withdrawn", not record["withdrawn"]),
                        (
                            "version",
                            max(row["version"] for row in facts["records"]) + 1,
                        ),
                    ]
                direct = False
                for field, value in changes:
                    if record[field] == value:
                        continue
                    altered = copy.deepcopy(case)
                    changed = altered["facts"]["records"][index]
                    changed[field] = value
                    if op == "choice_latest" and changed["issued"] > changed["expires"]:
                        continue
                    if base.oracle(altered) != original:
                        direct = True
                        break
                if not direct:
                    raise ValueError(
                        f"Choice record has no direct decision effect: {case['id']}"
                    )
                choice_records += 1
        elif op == "noul_all":
            witness = copy.deepcopy(case)
            for row in witness["facts"]["people"]:
                row.update({"cleared": True, "exempt": False, "suspended": False})
            if not base.oracle(witness):
                raise ValueError("Cannot make a positive roster witness")
            for index in range(len(facts["people"])):
                altered = copy.deepcopy(witness)
                altered["facts"]["people"][index]["suspended"] = True
                if base.oracle(altered):
                    raise ValueError(
                        "Roster entry is not necessary in positive witness"
                    )
                roster_records += 1
        elif op == "noul_chain":
            witness = copy.deepcopy(case)
            for row in witness["facts"]["links"]:
                row.update(
                    {"approved": True, "withdrawn": False, "expires": facts["day"]}
                )
            if not base.oracle(witness):
                raise ValueError("Cannot make a positive chain witness")
            for index in range(len(facts["links"])):
                altered = copy.deepcopy(witness)
                altered["facts"]["links"][index]["withdrawn"] = True
                if base.oracle(altered):
                    raise ValueError("Chain link is not necessary in positive witness")
                chain_links += 1
        elif op == "noul_revoke":
            for index, day in enumerate(facts["revocations"]):
                altered = copy.deepcopy(case)
                altered["facts"]["revocations"][index] = (
                    facts["day"] if day > facts["day"] else facts["day"] + 1
                )
                if base.oracle(altered) == original:
                    raise ValueError("Revocation day has no decision effect")
                revocation_dates += 1
        elif op == "score_weighted":
            for index, mark in enumerate(facts["marks"]):
                altered = copy.deepcopy(case)
                altered["facts"]["marks"][index] = 1 - mark
                if base.oracle(altered) == original:
                    raise ValueError("Weighted mark has no decision effect")
                weighted_marks += 1
        elif op == "score_penalty":
            for field in ("late", "open", "credit"):
                if not any(
                    base.oracle({**case, "facts": {**facts, field: value}}) != original
                    for value in range(5)
                    if value != facts[field]
                ):
                    raise ValueError("Penalty field has no decision effect")
                penalty_fields += 1
        elif op == "score_order":
            witness = copy.deepcopy(case)
            witness["facts"]["stages"] = [True, True, True]
            for index in range(3):
                altered = copy.deepcopy(witness)
                altered["facts"]["stages"][index] = False
                if base.oracle(altered) == base.oracle(witness):
                    raise ValueError(
                        "Ordered stage is not necessary in positive witness"
                    )
                order_stages += 1
    return {
        "choice_records_direct": choice_records,
        "roster_records_conditional": roster_records,
        "chain_links_conditional": chain_links,
        "revocation_dates_direct": revocation_dates,
        "weighted_marks_direct": weighted_marks,
        "penalty_fields_direct": penalty_fields,
        "ordered_stages_conditional": order_stages,
        "limitation": "Conditional witnesses do not imply each entry directly changes an already-failed conjunction",
    }


def make_rows(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    prompts, targets = [], []
    for case in cases:
        answer = base.oracle(case)
        for language in LANGUAGES:
            prompt = render(case, language)
            target = {
                "id": prompt["id"],
                "base_id": case["id"],
                "language": language,
                "task_type": KIND[case["operation"]],
                "operation": case["operation"],
                "semantic_gold": answer,
                "source_input_sha256": digest(
                    {"state": prompt["state"], "questions": prompt["questions"]}
                ),
            }
            if target["task_type"] == "choice":
                target["gold"] = chr(97 + case["option_order"].index(answer))
                target["semantic_by_label"] = {
                    chr(97 + i): item for i, item in enumerate(case["option_order"])
                }
            else:
                target["gold"] = answer
            prompts.append(prompt)
            targets.append(target)
    if len(prompts) != 72 or len({p["id"] for p in prompts}) != 72:
        raise ValueError("r7 expects 72 unique localized rows")
    for language in ("zh", "ja"):
        expected = "han" if language == "zh" else "kana"
        if any(
            sum(script_of(ch) == expected for ch in row["state"]) < 8
            for row in prompts
            if row["id"].endswith("-" + language)
        ):
            raise ValueError(f"{language}: locale script absent")
    return prompts, targets


def build(casebook: Path, output: Path, references: list[Path], revision: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin exact signed source revision")
    cases = base.load_cases(casebook)
    qa = casebook_qa(cases)
    prompts, targets = make_rows(cases)
    overlap = base.screen_overlap(prompts, references)
    if overlap["exact_matches"] or overlap["near_matches"]:
        raise ValueError("Protected/visible source overlap; quarantine candidate")
    output.mkdir(mode=0o700, parents=True)
    base._write_jsonl(output / "prompts.jsonl", prompts)
    base._write_jsonl(output / "targets.private.jsonl", targets)
    review_paths = {}
    for language in LANGUAGES:
        review = [
            {
                "id": prompt["id"],
                "base_id": target["base_id"],
                "language": language,
                "task_type": target["task_type"],
                "state": prompt["state"],
                "question": prompt["questions"]["decision"],
            }
            for prompt, target in zip(prompts, targets, strict=True)
            if target["language"] == language
        ]
        if len(review) != 18:
            raise ValueError("Locale packet missing a base")
        name = f"review.gold-free.{language}.jsonl"
        base._write_jsonl(output / name, review)
        review_paths[name] = sha256(output / name)
    manifest = {
        "schema_version": VERSION,
        "scope": "private DEV editorial candidate; no training or sealed FINAL",
        "source_revision": revision,
        "generator_sha256": sha256(Path(__file__)),
        "base_oracle_sha256": sha256(Path(base.__file__)),
        "native_preflight_sha256": sha256(
            Path(__file__).with_name("hard_native_preflight.py")
        ),
        "native_interpreter_sha256": sha256(Path(__file__).with_name("score.py")),
        "casebook_sha256": sha256(casebook),
        "private_targets_sha256": sha256(output / "targets.private.jsonl"),
        "files_sha256": {
            "prompts.jsonl": sha256(output / "prompts.jsonl"),
            **review_paths,
        },
        "overlap_screen": overlap,
        "automated_qa": qa,
        "counts": {
            "rows": 72,
            "independent_bases": 18,
            "by_type": {"choice": 24, "noul": 24, "score": 24},
            "by_language": dict.fromkeys(LANGUAGES, 18),
        },
        "inference_eligible": False,
        "training_approved": False,
        "publication_eligible": False,
        "independent_bilingual_review": "pending",
        "limitations": [
            "Mechanical correctness does not establish locale naturalness or translation fidelity",
            "Qualified native/bilingual blind review is required separately for every non-English locale",
            "Exact/near source screening does not establish semantic or sealed-FINAL decontamination",
            "Small DEV panel is not a release benchmark",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "manifest.json").chmod(0o600)
    return {"manifest_sha256": sha256(output / "manifest.json"), "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.casebook, args.output, args.reference, args.source_revision),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
