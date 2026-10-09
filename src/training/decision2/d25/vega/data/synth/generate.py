"""SYN1 generation: DESIGN -> INSTANCES -> assembled rows -> blind verification -> accepted rows.

Seeds run as independent async pipelines (bounded in flight) so the server stays saturated; results
are grouped into fixed shards of the plan. A shard directory holds an append-only request cache
(restarts replay finished calls), and is sealed by ``rows.jsonl.gz`` + ``rejects.jsonl.gz`` +
``stats.json`` + ``DONE``. Finished shards are skipped.

    python -m d25.vega.data.synth.generate --plan PLAN --work /data/d25/vega/synth/syn1 \
        --shards 0:20 --gen-url http://d25-vega-synth-llm:8000 --gen-model qwen3.5-397b-a17b-fp8 \
        --tokenizer /data/d25/shared/models/qwen3.5-397b-a17b-fp8
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data import util
from d25.vega.data.synth import prompts as P
from d25.vega.data.synth import taxonomy as tx
from d25.vega.data.synth.llm import LLM, RequestCache, parse_json
from d25.vega.data.synth.plan import KEY_STYLES

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
NON_CJK_WORD = re.compile(r"[^\W\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
CATCHALL = re.compile(
    r"(?i)\b(none|other|others|not listed|no match|neither|not applicable|n/a|unknown|insufficient|"
    r"cannot be determined|can't be determined|not stated|not enough|unclear|undetermined|no suitable|"
    r"not specified|keine|aucun|ninguno|nenhum|nessuno|geen)\b"
)
LETTERS = [chr(ord("A") + i) for i in range(26)]
STATE_WORD = re.compile(r"(?i)\b(state|record below|attached|provided)\b")
ERROR_REASON = re.compile(
    r"^(design_error|instances_error|verify_error|tier1_error|seed_exception)|Error\("
)
REPAIRABLE = {
    "author_mismatch",
    "label_disagreement",
    "ambiguous",
    "answer_leak",
    "unnatural",
}
WORD_BOUNDS = {
    "clean": (40, 150),
    "noisy": (40, 150),
    "negated": (40, 150),
    "long": (300, 600),
    "shifted_prior": (40, 200),
}
NOTA_ELIGIBLE = {
    "criteria_classification",
    "routing",
    "knowledge_application",
    "long_state_lookup",
    "temporal_numeric",
    "eligibility",
    "pii_sensitivity",
    "agent_trace",
}
NOTA_TEXTS = (
    ("none_of_the_above", "None of the other options applies."),
    ("None of these", "No listed option fits this case."),
    ("not_listed", "The correct answer is not among the other options."),
    ("none", "None of the above."),
)
FAMILY_RULES = {
    "pair": (
        "Two variants of the same scenario. Keep people, setting and format; change only the "
        "decisive fact(s) so that each requested answer is uniquely correct."
    ),
    "skew": (
        "Four variants of the same scenario type. Variants that share a requested answer are "
        "distinct cases (different names, amounts or details that do not change the answer), "
        "not paraphrases; the odd one out differs in the decisive fact."
    ),
    "multi": (
        "Two or three variants of the same scenario, each a rich state that supports every "
        "question; change a few facts between variants so that answers differ."
    ),
}


def words(state: Any) -> float:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return len(NON_CJK_WORD.findall(text)) + len(CJK.findall(text)) / 1.7


def rng_for(*parts: object) -> random.Random:
    return random.Random(
        int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)
    )


def req_seed(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def is_catchall(key: str, description: str | None) -> bool:
    return bool(CATCHALL.search(f"{key} {description or ''}"))


def bounds(seed: dict, condition: str) -> tuple[int, int]:
    low, high = WORD_BOUNDS[condition]
    if seed["kind"] == "multi":
        low, high = int(low * 1.6), int(high * 1.4)
    return low, high


class Pipeline:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.gen = LLM(cfg.gen_url, cfg.gen_model, concurrency=cfg.concurrency)
        self.ver = (
            self.gen
            if (cfg.ver_url == cfg.gen_url and cfg.ver_model == cfg.gen_model)
            else LLM(cfg.ver_url, cfg.ver_model, concurrency=cfg.concurrency)
        )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self.codes, self.code_ids = df.answer_codes(self.tokenizer)
        self.teacher = cfg.ver_model
        self.counters: Counter = Counter()
        self.prompts_sha = P.prompts_sha()

    # ---------------------------------------------------------------- DESIGN
    def design_messages(self, seed: dict) -> list[dict]:
        spec = tx.ARCHETYPES[seed["archetype"]]
        kind = seed["kind"]
        request: dict[str, Any] = {
            "domain": seed["domain"],
            "sector": seed["sector"].replace("_", " "),
            "archetype": seed["archetype"],
            "archetype_description": spec["description"],
            "focus": seed["phenomenon"],
            "question_type": {
                "choice": "choice (pick one option)",
                "noul": "yes/no",
                "multi": "3-5 questions about one state, mixing choice and yes/no",
            }[kind],
            "question_language": seed["question_language"],
            "state_language": seed["state_language"],
            "state_format": (
                "JSON object" if seed["state_format"] == "json" else "plain text"
            ),
        }
        notes = []
        if kind == "choice":
            request["number_of_options"] = seed["n_options"]
            request["key_style"] = KEY_STYLES[seed["key_style"]]
            if seed["ordinal"]:
                notes.append(
                    "The options form an ordered scale; list them from lowest to highest."
                )
            if seed["archetype"] == "none_of_the_above":
                notes.append(
                    "Exactly one option is the explicit 'none of these' option; put it last."
                )
            if seed["n_options"] > 30:
                notes.append(
                    "With many options keep each description to one short distinct line."
                )
        if kind == "multi":
            notes.append(
                "Choice questions have 2-10 options; noul questions leave options empty. "
                "Fill yes_means/no_means only for yes/no questions (empty strings otherwise)."
            )
        if notes:
            request["notes"] = notes
        return [
            {"role": "system", "content": P.DESIGN_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(request, ensure_ascii=False, indent=1),
            },
        ]

    def check_design(self, seed: dict, design: dict) -> str | None:
        kind = seed["kind"]
        if (
            kind in ("choice", "noul")
            and not str(design.get("instructions", "")).strip()
        ):
            return "no_instructions"
        if kind == "choice":
            options = design["options"]
            keys = [str(o["key"]).strip() for o in options]
            if len(options) != seed["n_options"]:
                return "option_count"
            if any(not k or len(k) > 80 for k in keys):
                return "bad_key"
            if len({k.casefold() for k in keys}) != len(keys):
                return "duplicate_key"
            catchalls = sum(is_catchall(o["key"], o["description"]) for o in options)
            if seed["archetype"] == "none_of_the_above" and catchalls < 1:
                return "nota_missing"
        elif kind == "noul":
            if not design["yes_means"].strip() or not design["no_means"].strip():
                return "no_yes_no_means"
        else:
            questions = design["questions"]
            qids = [q["qid"].strip() for q in questions]
            if len(set(qids)) != len(qids) or any(not q for q in qids):
                return "bad_qids"
            for q in questions:
                if not q["instructions"].strip():
                    return "no_instructions"
                if q["type"] == "choice":
                    keys = [str(o["key"]).strip() for o in q["options"]]
                    if (
                        not 2 <= len(keys) <= 10
                        or len({k.casefold() for k in keys}) != len(keys)
                        or not all(keys)
                    ):
                        return "bad_multi_options"
                elif not q["yes_means"].strip() or not q["no_means"].strip():
                    return "no_yes_no_means"
        return None

    async def design(self, seed: dict, cache: RequestCache) -> tuple[dict | None, str]:
        schema = P.design_schema(seed["kind"], seed["n_options"] or None)
        max_tokens = 2500 + (40 * seed["n_options"] if seed["kind"] == "choice" else 0)
        reason = "design_failed"
        for attempt in range(3):
            result = await self.gen.chat(
                self.design_messages(seed),
                cache=cache,
                schema=schema,
                thinking=self.cfg.think_design,
                budget=self.cfg.design_budget,
                max_tokens=max_tokens
                + (self.cfg.design_budget if self.cfg.think_design else 0),
                temperature=0.85,
                top_p=0.95,
                top_k=50,
                seed=req_seed(seed["gen_seed"], "design", attempt),
            )
            if "error" in result:
                reason = f"design_error:{result['error'].get('status')}"
                continue
            if result["finish"] != "stop":
                reason = "design_truncated"
                continue
            try:
                design = parse_json(result["content"])
            except (json.JSONDecodeError, ValueError):
                reason = "design_json"
                continue
            problem = self.check_design(seed, design)
            if problem is None:
                design["_attempt"] = attempt
                return design, "ok"
            reason = f"design_{problem}"
        return None, reason

    # ------------------------------------------------------------- INSTANCES
    def wanted(self, seed: dict, design: dict, condition: str) -> list[str]:
        planned = seed["answers"][condition]
        if seed["kind"] == "choice":
            return [str(design["options"][i]["key"]).strip() for i in planned]
        return list(planned)

    def instance_messages(
        self,
        seed: dict,
        design: dict,
        condition: str,
        ids: list[str],
        feedback: list[dict] | None = None,
    ) -> list[dict]:
        kind = seed["kind"]
        task: dict[str, Any] = {
            "setting": design["setting"],
            "shared_context": design["shared_context"],
            "decisive_factors": design["decisive_factors"],
        }
        if kind == "choice":
            task["instructions"] = design["instructions"]
            task["options"] = [
                {"key": str(o["key"]).strip(), "description": o["description"]}
                for o in design["options"]
            ]
        elif kind == "noul":
            task["instructions"] = design["instructions"]
            task["yes_means"] = design["yes_means"]
            task["no_means"] = design["no_means"]
        else:
            task["questions"] = [
                {
                    k: v
                    for k, v in q.items()
                    if (q["type"] == "choice" and k != "yes_means" and k != "no_means")
                    or (q["type"] == "noul" and k != "options")
                }
                for q in design["questions"]
            ]
        low, high = bounds(seed, condition)
        layout = "multi" if kind == "multi" else tx.CONDITION_LAYOUT[condition][1]
        if kind == "multi":
            variants = [{"id": i} for i in ids]
        else:
            variants = [
                {"id": i, "requested_answer": w}
                for i, w in zip(ids, self.wanted(seed, design, condition))
            ]
        request = {
            "task": task,
            "archetype": seed["archetype"],
            "archetype_description": tx.ARCHETYPES[seed["archetype"]]["description"],
            "focus": seed["phenomenon"],
            "condition": condition,
            "condition_description": tx.CONDITIONS[condition],
            "style": seed["styles"][condition],
            "state_format": (
                "JSON object" if seed["state_format"] == "json" else "plain text"
            ),
            "state_language": seed["state_language"],
            "word_bounds_per_state": [low, high],
            "family_rule": FAMILY_RULES[layout],
            "variants": variants,
        }
        if seed["state_language"] in ("Chinese", "Japanese"):
            request["note"] = (
                "For Chinese or Japanese text, count about 1.7 characters as one word."
            )
        if feedback:
            request["validation_feedback"] = feedback
        system = P.INSTANCE_SYSTEM_MULTI if kind == "multi" else P.INSTANCE_SYSTEM
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(request, ensure_ascii=False, indent=1),
            },
        ]

    async def instances(
        self,
        seed: dict,
        design: dict,
        condition: str,
        cache: RequestCache,
        feedback: list[dict] | None = None,
    ):
        kind = seed["kind"]
        salt = "repair" if feedback else ""
        if kind == "multi":
            count = 2 if condition != "long" else 2
            if condition == "shifted_prior":
                count = 3
        else:
            count = tx.CONDITION_LAYOUT[condition][0]
        ids = [f"v{i}" for i in range(count)]
        if kind == "choice":
            answers: Any = [str(o["key"]).strip() for o in design["options"]]
        elif kind == "noul":
            answers = ["yes", "no"]
        else:
            answers = {
                q["qid"].strip(): (
                    [str(o["key"]).strip() for o in q["options"]]
                    if q["type"] == "choice"
                    else ["yes", "no"]
                )
                for q in design["questions"]
            }
        schema = P.instance_schema(ids, answers, seed["state_format"] == "json")
        per_variant = 1500 if condition == "long" else 650
        if seed["state_format"] == "json":
            per_variant = int(per_variant * 1.5)
        if kind == "multi":
            per_variant = int(per_variant * 1.4)
        max_tokens = min(1500 + count * per_variant, 14000)
        messages = self.instance_messages(seed, design, condition, ids, feedback)
        reason = "instances_failed"
        for attempt in range(2):
            result = await self.gen.chat(
                messages,
                cache=cache,
                schema=schema,
                thinking=self.cfg.think_instances,
                budget=self.cfg.instance_budget,
                max_tokens=max_tokens
                + (self.cfg.instance_budget if self.cfg.think_instances else 0),
                temperature=0.85,
                top_p=0.95,
                top_k=50,
                seed=req_seed(seed["gen_seed"], condition, attempt, salt),
            )
            if "error" in result:
                reason = f"instances_error:{result['error'].get('status')}"
                continue
            if result["finish"] != "stop":
                reason = "instances_truncated"
                continue
            try:
                out = parse_json(result["content"])
            except (json.JSONDecodeError, ValueError):
                reason = "instances_json"
                continue
            by_id = {v["id"]: v for v in out.get("variants", [])}
            if set(by_id) != set(ids):
                reason = "instances_ids"
                continue
            return {
                "plan": out.get("plan", ""),
                "variants": [by_id[i] for i in ids],
            }, "ok"
        return None, reason

    # -------------------------------------------------------------- ASSEMBLY
    def render_choice(
        self,
        rng: random.Random,
        options: list[tuple[str, str]],
        ordinal: bool,
        instructions: str,
    ) -> tuple[dict, dict[str, str], str]:
        order = list(range(len(options)))
        if not ordinal:
            rng.shuffle(order)
        style = rng.choices(
            ["semantic", "option_index", "letters"], [0.70, 0.18, 0.12]
        )[0]
        if style == "letters" and len(options) > 26:
            style = "option_index"
        criteria: dict[str, Any] = {}
        key_map: dict[str, str] = {}
        for pos, index in enumerate(order):
            key, desc = options[index]
            if style == "semantic":
                final_key, final_desc = key, (desc or None)
            else:
                final_key = f"option_{pos}" if style == "option_index" else LETTERS[pos]
                final_desc = f"{key}: {desc}" if desc else key
            criteria[final_key] = final_desc
            key_map[key] = final_key
        return (
            {"type": "choice", "instructions": instructions, "criteria": criteria},
            key_map,
            style,
        )

    def render_noul(
        self, rng: random.Random, instructions: str, yes_means: str, no_means: str
    ):
        form = rng.choices(
            ["noul", "noul_criteria", "choice_yes_no"], [0.55, 0.15, 0.30]
        )[0]
        if form == "noul":
            return (
                {"type": "noul", "instructions": instructions},
                {"yes": "true", "no": "false"},
                form,
            )
        if form == "noul_criteria":
            return (
                {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": {"false": no_means, "true": yes_means},
                },
                {"yes": "true", "no": "false"},
                form,
            )
        yes_key, no_key = rng.choice([("yes", "no"), ("Yes", "No")])
        pairs = [(yes_key, yes_means), (no_key, no_means)]
        rng.shuffle(pairs)
        return (
            {"type": "choice", "instructions": instructions, "criteria": dict(pairs)},
            {"yes": yes_key, "no": no_key},
            form,
        )

    def assemble(
        self, seed: dict, design: dict, condition: str, inst: dict
    ) -> list[dict]:
        rng = rng_for(seed["seed_id"], condition, "assemble")
        context = str(design.get("shared_context") or "").strip()
        placement = (
            None if not context else ("state" if rng.random() < 0.6 else "instructions")
        )
        content_in_instructions = (
            seed["state_format"] == "text"
            and condition != "long"
            and seed["kind"] != "multi"
            and rng.random() < 0.08
            and not STATE_WORD.search(str(design.get("instructions", "")))
        )
        if seed["kind"] == "multi":
            questions = [(q["qid"].strip(), q) for q in design["questions"]]
        else:
            questions = [("q", design)]
        wanted = (
            self.wanted(seed, design, condition) if seed["kind"] != "multi" else None
        )
        rows = []
        renders: dict[str, tuple] = {}
        for qid, q in questions:
            instructions = str(q["instructions"]).strip()
            if placement == "instructions":
                instructions = f"{instructions}\n\n{context}"
            qtype = q["type"] if seed["kind"] == "multi" else seed["kind"]
            if qtype == "choice":
                options = [
                    (str(o["key"]).strip(), str(o["description"] or "").strip())
                    for o in q["options"]
                ]
                ordinal = seed["ordinal"] if seed["kind"] != "multi" else False
                renders[qid] = (
                    ("choice",)
                    + self.render_choice(rng, options, ordinal, instructions)
                    + (options,)
                )
            else:
                renders[qid] = (
                    ("noul",)
                    + self.render_noul(
                        rng, instructions, q["yes_means"].strip(), q["no_means"].strip()
                    )
                    + (None,)
                )
        low, high = bounds(seed, condition)
        for v_index, variant in enumerate(inst["variants"]):
            state = variant["state"]
            if isinstance(state, str):
                state = state.strip()
            n_words = words(state)
            if placement == "state":
                state = (
                    f"{context}\n\n{state}"
                    if isinstance(state, str)
                    else {"reference": context, "record": state}
                )
            for qid, q in questions:
                kind, question, key_map, form, options = renders[qid]
                question = json.loads(json.dumps(question))
                row_state = state
                if content_in_instructions and isinstance(variant["state"], str):
                    question["instructions"] = (
                        f"{question['instructions']}\n\n{variant['state'].strip()}"
                    )
                    row_state = context if placement == "state" else ""
                if seed["kind"] == "multi":
                    author = str(variant["answers"].get(qid, "")).strip()
                    planned = author
                else:
                    author = str(variant["answer"]).strip()
                    planned = wanted[v_index]
                final_key = key_map.get(planned)
                if question["type"] == "noul":
                    gold_index = 1 if final_key == "true" else 0
                else:
                    gold_index = (
                        list(question["criteria"]).index(final_key)
                        if final_key in question["criteria"]
                        else -1
                    )
                row_id = f"syn1-{seed['seed_id']}-{condition}-{variant['id']}" + (
                    f"-{qid}" if seed["kind"] == "multi" else ""
                )
                rows.append(
                    {
                        "id": row_id,
                        "state": row_state,
                        "question": question,
                        "gold_index": gold_index,
                        "author_ok": author == planned and gold_index >= 0,
                        "author": author,
                        "planned": planned,
                        "explanation": variant.get("explanation", ""),
                        "words": round(n_words, 1),
                        "length_ok": low * 0.6 <= n_words <= high * 1.6,
                        "meta": {
                            "seed_id": seed["seed_id"],
                            "group": f"{seed['seed_id']}-{condition}",
                            "variant": v_index,
                            "qid": qid if seed["kind"] == "multi" else None,
                            "condition": condition,
                            "sector": seed["sector"],
                            "domain": seed["domain"],
                            "archetype": seed["archetype"],
                            "phenomenon": seed["phenomenon"],
                            "kind": seed["kind"],
                            "question_form": form,
                            "context_placement": placement,
                            "content_in_instructions": content_in_instructions,
                            "state_format": seed["state_format"],
                            "style": seed["styles"][condition],
                            "state_language": seed["state_language"],
                            "question_language": seed["question_language"],
                            "nota": None,
                            "repaired": False,
                        },
                        "_options": options,
                        "_key_map": key_map,
                    }
                )
        return rows

    def nota_pairs(self, seed: dict, rows: list[dict]) -> list[dict]:
        if (
            seed["kind"] != "choice"
            or seed["archetype"] not in NOTA_ELIGIBLE
            or seed["ordinal"]
        ):
            return []
        out = []
        for row in rows:
            options = row["_options"]
            if (
                not row["author_ok"]
                or row["meta"]["question_form"]
                not in ("semantic", "option_index", "letters")
                or not 3 <= len(options) <= 60
                or any(is_catchall(k, d) for k, d in options)
            ):
                continue
            rng = rng_for(row["id"], "nota")
            if rng.random() >= self.cfg.nota_rate:
                continue
            nota_key, nota_desc = rng.choice(NOTA_TEXTS)
            if any(nota_key.casefold() == k.casefold() for k, _ in options):
                continue
            gold_key = row["planned"]
            style = row["meta"]["question_form"]
            for variant, keep_gold in (("nota1", True), ("nota0", False)):
                items = [
                    (k, d)
                    for k, d in self.ordered_options(row)
                    if keep_gold or k != gold_key
                ]
                position = (
                    len(items) if rng.random() < 0.75 else rng.randrange(len(items) + 1)
                )
                items.insert(position, (nota_key, nota_desc))
                criteria, final_gold = {}, None
                for pos, (key, desc) in enumerate(items):
                    if style == "semantic":
                        final_key, final_desc = key, (desc or None)
                    else:
                        final_key = (
                            f"option_{pos}"
                            if (style == "option_index" or len(items) > 26)
                            else LETTERS[pos]
                        )
                        final_desc = f"{key}: {desc}" if desc else key
                    criteria[final_key] = final_desc
                    if key == (gold_key if keep_gold else nota_key):
                        final_gold = final_key
                question = dict(row["question"], criteria=criteria)
                new = dict(
                    row,
                    id=f"{row['id']}-{variant}",
                    question=question,
                    gold_index=list(criteria).index(final_gold),
                    meta=dict(
                        row["meta"],
                        nota="gold_present" if keep_gold else "gold_removed",
                    ),
                )
                out.append(new)
        return out

    @staticmethod
    def ordered_options(row: dict) -> list[tuple[str, str]]:
        """Original (key, description) pairs in the row's displayed order."""
        inverse = {final: original for original, final in row["_key_map"].items()}
        lookup = dict(row["_options"])
        return [(inverse[k], lookup[inverse[k]]) for k in row["question"]["criteria"]]

    # ---------------------------------------------------------- VERIFICATION
    async def verify(self, row: dict, cache: RequestCache, salt: str = "") -> dict:
        question = row["question"]
        labels = P.labels_for(question)
        messages = [
            {"role": "system", "content": P.VERIFY_SYSTEM},
            {"role": "user", "content": P.render_item(row["state"], question)},
        ]
        result = await self.ver.chat(
            messages,
            cache=cache,
            schema=P.verify_schema(labels),
            thinking=True,
            budget=self.cfg.verify_budget,
            max_tokens=self.cfg.verify_budget + 1200,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=req_seed(row["id"], "verify", salt),
        )
        if "error" in result or result["finish"] != "stop":
            return {
                "ok": False,
                "reason": "verify_error" if "error" in result else "verify_truncated",
            }
        try:
            judgment = parse_json(result["content"])
        except (json.JSONDecodeError, ValueError):
            return {"ok": False, "reason": "verify_json"}
        label = judgment.get("label")
        index = labels.index(label) if label in labels else -1
        if question["type"] == "noul":
            index = {"no": 0, "yes": 1}.get(label, -1)
        return {
            "ok": True,
            "index": index,
            "label": label,
            **{
                k: judgment.get(k)
                for k in (
                    "explanation",
                    "ambiguous",
                    "answer_leak",
                    "unnatural",
                    "language",
                    "quality_issue",
                )
            },
            "reasoning_tokens": result["usage"]["completion"],
        }

    async def tier1(self, row: dict, cache: RequestCache) -> dict:
        n = len(df.options(row["question"])[0])
        prompt = df.render(self.tokenizer, row["state"], row["question"], self.codes)
        return await self.ver.code_probs(prompt, self.code_ids[:n], cache=cache)

    async def check_row(self, row: dict, cache: RequestCache) -> dict:
        reasons = []
        if not row["author_ok"]:
            return dict(row, reasons=["author_mismatch"])
        blind, probs = await asyncio.gather(
            self.verify(row, cache), self.tier1(row, cache)
        )
        row = dict(row, blind=blind, tier1=probs)
        if not blind.get("ok"):
            reasons.append(blind.get("reason", "verify_failed"))
        else:
            if blind["index"] != row["gold_index"]:
                reasons.append("label_disagreement")
            for flag in ("ambiguous", "answer_leak", "unnatural"):
                if blind.get(flag) is not False:
                    reasons.append(flag)
        if "error" in probs:
            reasons.append("tier1_error")
        elif (
            not reasons
            and probs["probs"][row["gold_index"]] < self.cfg.second_opinion_below
        ):
            second = await self.verify(row, cache, salt="second")
            row["second"] = second
            if (
                not second.get("ok")
                or second["index"] != row["gold_index"]
                or second.get("ambiguous") is not False
            ):
                reasons.append("second_opinion")
        return dict(row, reasons=reasons)

    # ------------------------------------------------------------- PER SEED
    async def check_group(
        self, seed: dict, design: dict, condition: str, inst: dict, cache: RequestCache
    ) -> list[dict]:
        rows = self.assemble(seed, design, condition, inst)
        rows += self.nota_pairs(seed, rows)
        return list(await asyncio.gather(*(self.check_row(r, cache) for r in rows)))

    @staticmethod
    def feedback_for(checked: list[dict]) -> list[dict]:
        notes: dict[str, dict] = {}
        for row in checked:
            if row["meta"]["nota"] is not None or not row["reasons"]:
                continue
            vid = f"v{row['meta']['variant']}"
            entry = notes.setdefault(
                vid, {"id": vid, "problems": [], "reviewer_notes": []}
            )
            entry["problems"] = sorted(set(entry["problems"]) | set(row["reasons"]))
            blind = row.get("blind") or {}
            if "author_mismatch" in row["reasons"]:
                entry["reviewer_notes"].append(
                    f"Your answer was {row['author']!r} but the requested answer is {row['planned']!r}."
                )
            if blind.get("quality_issue"):
                entry["reviewer_notes"].append(str(blind["quality_issue"])[:400])
            if (
                "label_disagreement" in row["reasons"]
                and blind.get("label") is not None
            ):
                entry["reviewer_notes"].append(
                    f"An independent reviewer chose {blind['label']!r}: {str(blind.get('explanation', ''))[:300]}"
                )
        return list(notes.values())

    async def run_group(
        self, seed: dict, design: dict, condition: str, cache: RequestCache
    ):
        inst, status = await self.instances(seed, design, condition, cache)
        if inst is None:
            return None, status
        checked = await self.check_group(seed, design, condition, inst, cache)
        self.counters["candidates_first_pass"] += len(checked)
        self.counters["accepted_first_pass"] += sum(not r["reasons"] for r in checked)
        needs = any(
            r["meta"]["nota"] is None
            and r["reasons"]
            and set(r["reasons"]) <= REPAIRABLE
            for r in checked
        )
        if self.cfg.repair and needs:
            self.counters["repair_attempts"] += 1
            inst2, _ = await self.instances(
                seed, design, condition, cache, feedback=self.feedback_for(checked)
            )
            if inst2 is not None:
                checked2 = await self.check_group(seed, design, condition, inst2, cache)
                if sum(not r["reasons"] for r in checked2) > sum(
                    not r["reasons"] for r in checked
                ):
                    self.counters["repair_improved"] += 1
                    for row in checked2:
                        row["meta"]["repaired"] = True
                    checked = checked2
        return checked, "ok"

    async def run_seed(
        self, seed: dict, cache: RequestCache
    ) -> tuple[list[dict], list[dict]]:
        design, status = await self.design(seed, cache)
        if design is None:
            self.counters[status] += 1
            return [], [
                {"seed_id": seed["seed_id"], "stage": "design", "reason": status}
            ]
        conditions = list(tx.CONDITIONS)
        results = await asyncio.gather(
            *(self.run_group(seed, design, c, cache) for c in conditions)
        )
        accepted, rejects = [], []
        for condition, (checked, status) in zip(conditions, results):
            if checked is None:
                self.counters[status] += 1
                rejects.append(
                    {
                        "seed_id": seed["seed_id"],
                        "stage": "instances",
                        "condition": condition,
                        "reason": status,
                    }
                )
                continue
            for row in checked:
                self.counters["candidates"] += 1
                if row["reasons"]:
                    self.counters.update(row["reasons"])
                    rejects.append(self.reject_record(seed, row))
                else:
                    self.counters["accepted"] += 1
                    accepted.append(self.final_row(seed, design, row))
        return accepted, rejects

    def reject_record(self, seed: dict, row: dict) -> dict:
        blind = row.get("blind") or {}
        return {
            "seed_id": seed["seed_id"],
            "stage": "verify",
            "id": row["id"],
            "reasons": row["reasons"],
            "state": row["state"],
            "question": row["question"],
            "gold_index": row["gold_index"],
            "author": row["author"],
            "planned": row["planned"],
            "blind_index": blind.get("index"),
            "blind_issue": blind.get("quality_issue"),
            "blind_explanation": blind.get("explanation"),
            "tier1": (row.get("tier1") or {}).get("probs"),
            "meta": row["meta"],
        }

    def final_row(self, seed: dict, design: dict, row: dict) -> dict:
        n = len(df.options(row["question"])[0])
        target = [0.0] * n
        target[row["gold_index"]] = 1.0
        blind = row["blind"]
        meta = dict(row["meta"])
        meta.update(
            {
                "licence": "synthetic",
                "split": "train",
                "corpus": "SYN1",
                "generator": {
                    "model": self.cfg.gen_repo,
                    "revision": self.cfg.gen_revision,
                    "served": self.cfg.gen_model,
                    "design_thinking": self.cfg.think_design,
                    "instance_thinking": self.cfg.think_instances,
                },
                "verifier": {
                    "model": self.cfg.ver_repo,
                    "revision": self.cfg.ver_revision,
                    "served": self.cfg.ver_model,
                    "thinking_budget": self.cfg.verify_budget,
                },
                "prompts": {"version": P.VERSION, "sha": self.prompts_sha},
                "verification": {
                    "author": row["author"],
                    "blind": blind.get("label"),
                    "agree": True,
                    "ambiguous": blind.get("ambiguous"),
                    "answer_leak": blind.get("answer_leak"),
                    "unnatural": blind.get("unnatural"),
                    "language": blind.get("language"),
                    "second_opinion": (row.get("second") or {}).get("label"),
                    "explanation": blind.get("explanation"),
                    "author_explanation": row["explanation"],
                },
                "words": row["words"],
                "length_ok": row["length_ok"],
                "gold_target": target,
                "teachers": {
                    self.teacher: [round(p, 6) for p in row["tier1"]["probs"]]
                },
                "teacher_prompt": df.FORMAT_ID,
                "design_setting": design.get("setting"),
            }
        )
        return util.make_row(
            row_id=row["id"],
            source="syn1",
            family=f"syn1/{seed['archetype']}/{seed['seed_id']}",
            state=row["state"],
            question=row["question"],
            target=target,
            meta=meta,
        )


async def run(cfg: argparse.Namespace) -> None:
    seeds = list(util.read_jsonl(cfg.plan))
    shard_size = cfg.shard_size
    first, last = (int(x) for x in cfg.shards.split(":"))
    last = min(last, (len(seeds) + shard_size - 1) // shard_size)
    work = Path(cfg.work)
    pipeline = Pipeline(cfg)
    inflight = asyncio.Semaphore(cfg.inflight_seeds)
    started = time.time()

    async def seed_task(seed, cache):
        async with inflight:
            try:
                return await pipeline.run_seed(seed, cache)
            except Exception as error:  # keep the shard going; record the failure
                pipeline.counters["seed_exception"] += 1
                return [], [
                    {
                        "seed_id": seed["seed_id"],
                        "stage": "exception",
                        "reason": repr(error)[:500],
                    }
                ]

    async def shard_task(index: int):
        shard_dir = work / "shards" / f"{index:05d}"
        if (shard_dir / "DONE").exists():
            return
        shard_dir.mkdir(parents=True, exist_ok=True)
        cache = RequestCache(shard_dir / "cache.jsonl")
        chunk = seeds[index * shard_size : (index + 1) * shard_size]
        outcomes = await asyncio.gather(*(seed_task(s, cache) for s in chunk))
        accepted = [r for a, _ in outcomes for r in a]
        rejects = [r for _, rj in outcomes for r in rj]
        errored = {
            r["seed_id"]
            for r in rejects
            if any(
                ERROR_REASON.match(str(x))
                for x in (r.get("reasons") or [r.get("reason")])
            )
        }
        if len(errored) > max(2, 0.05 * len(chunk)):
            cache.close()
            print(
                f"shard {index} NOT sealed: {len(errored)} seeds hit server errors; rerun to resume",
                flush=True,
            )
            return
        util.write_jsonl(shard_dir / "rows.jsonl.gz", accepted)
        util.write_jsonl(shard_dir / "rejects.jsonl.gz", rejects)
        stats = {
            "shard": index,
            "seeds": len(chunk),
            "accepted": len(accepted),
            "rejects": dict(
                Counter(
                    r for x in rejects for r in (x.get("reasons") or [x.get("reason")])
                )
            ),
            "by_archetype": dict(Counter(r["meta"]["archetype"] for r in accepted)),
            "by_condition": dict(Counter(r["meta"]["condition"] for r in accepted)),
        }
        util.write_json(shard_dir / "stats.json", stats)
        cache.close()
        (shard_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        print(
            f"shard {index} done: {len(accepted)} accepted / {len(rejects)} rejects",
            flush=True,
        )

    async def reporter():
        while True:
            await asyncio.sleep(60)
            c = pipeline.counters
            rate = c["accepted"] / max(c["candidates"], 1)
            print(
                json.dumps(
                    {
                        "t": round(time.time() - started),
                        "accepted": c["accepted"],
                        "candidates": c["candidates"],
                        "accept_rate": round(rate, 3),
                        "top_rejects": c.most_common(8),
                        "gen": pipeline.gen.stats.snapshot(),
                    }
                ),
                flush=True,
            )

    report = asyncio.create_task(reporter())
    shard_ids = list(range(first, last))
    parallel = asyncio.Semaphore(cfg.parallel_shards)

    async def guarded(index):
        async with parallel:
            await shard_task(index)

    await asyncio.gather(*(guarded(i) for i in shard_ids))
    report.cancel()
    print(
        json.dumps(
            {"final": dict(pipeline.counters), "gen": pipeline.gen.stats.snapshot()}
        ),
        flush=True,
    )
    await pipeline.gen.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument(
        "--shards", default="0:1000000", help="first:last shard index (exclusive)"
    )
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--parallel-shards", type=int, default=8)
    ap.add_argument("--inflight-seeds", type=int, default=160)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--gen-url", required=True)
    ap.add_argument("--gen-model", required=True)
    ap.add_argument("--gen-repo", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument(
        "--gen-revision", default="ea5b4f81096f3901c91dea97f81324302495781d"
    )
    ap.add_argument("--ver-url")
    ap.add_argument("--ver-model")
    ap.add_argument("--ver-repo")
    ap.add_argument("--ver-revision")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--think-design", action="store_true")
    ap.add_argument("--think-instances", action="store_true")
    ap.add_argument("--design-budget", type=int, default=2048)
    ap.add_argument("--instance-budget", type=int, default=2048)
    ap.add_argument("--verify-budget", type=int, default=2048)
    ap.add_argument("--nota-rate", type=float, default=0.3)
    ap.add_argument("--second-opinion-below", type=float, default=0.1)
    ap.add_argument("--no-repair", dest="repair", action="store_false")
    cfg = ap.parse_args(argv)
    cfg.ver_url = cfg.ver_url or cfg.gen_url
    cfg.ver_model = cfg.ver_model or cfg.gen_model
    cfg.ver_repo = cfg.ver_repo or cfg.gen_repo
    cfg.ver_revision = cfg.ver_revision or cfg.gen_revision
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
