import collections
import json
import unittest

from training.model.data import validate_row
from v2.data.ib2 import audit, build, review
from v2.data.ib2 import families as fam


def counter():
    return collections.Counter()


SYSTEM = (
    "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
    + json.dumps(
        {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
        indent=4,
    )
    + "\n\n"
    + json.dumps(
        {"name": "get_news", "description": "Get the latest news", "parameters": {}}
    )
    + "\n"
)


def chat(*turns):
    return "".join(f"{role}: {text} <|endoftext|>\n\n\n" for role, text in turns)


def call(name, args):
    return '<functioncall> {"name": "%s", "arguments": \'%s\'}' % (
        name,
        json.dumps(args),
    )


class GlaiveParsingTest(unittest.TestCase):
    def test_functions_turns_and_calls(self):
        functions = fam.glaive_functions(SYSTEM)
        self.assertEqual(
            [f["name"] for f in functions], ["convert_currency", "get_news"]
        )
        self.assertIsNone(
            fam.glaive_functions("SYSTEM: no access to external functions")
        )
        text = chat(
            ("USER", "Convert 100 USD to EUR please."),
            (
                "ASSISTANT",
                call(
                    "convert_currency",
                    {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                ),
            ),
            ("FUNCTION RESPONSE", '{"converted_amount": 85}'),
            ("ASSISTANT", "100 USD is 85 EUR."),
        )
        turns = fam.glaive_turns(text)
        self.assertEqual(
            [t[0] for t in turns],
            ["USER", "ASSISTANT", "FUNCTION RESPONSE", "ASSISTANT"],
        )
        name, args = fam.glaive_call(turns[1][1], {"convert_currency", "get_news"})
        self.assertEqual((name, args["amount"]), ("convert_currency", 100))
        self.assertIsNone(fam.glaive_call(turns[1][1], {"get_news"}))

    def test_grounding_and_markers(self):
        request = fam.Request("Please convert 1,500 dollars to EUR for my two kids")
        self.assertTrue(request.grounded(1500))
        self.assertTrue(request.grounded(2))
        self.assertTrue(request.grounded("eur"))
        self.assertFalse(request.grounded("GBP"))
        self.assertFalse(request.grounded(True))
        self.assertFalse(request.grounded("s"))
        self.assertTrue(
            fam.is_refusal(
                "I'm sorry, but I don't have the capability to book flights."
            )
        )
        self.assertFalse(
            fam.is_refusal("I'm sorry, could you please provide the amount?")
        )
        self.assertTrue(fam.is_request("Sure, could you please provide the amount?"))
        self.assertEqual(fam.name_tokens("getStockPrice"), {"stock", "price"})
        self.assertEqual(fam.name_tokens("calculate_loan_payment"), {"loan", "payment"})


def conversations():
    rows = []
    for i in range(40):
        amount = 100 + i
        rows.append(
            {
                "system": SYSTEM,
                "chat": chat(
                    ("USER", f"Convert {amount} USD to EUR please, request {i}."),
                    (
                        "ASSISTANT",
                        call(
                            "convert_currency",
                            {
                                "amount": amount,
                                "from_currency": "USD",
                                "to_currency": "EUR",
                            },
                        ),
                    ),
                ),
            }
        )
        rows.append(
            {
                "system": SYSTEM,
                "chat": chat(
                    ("USER", f"Can you book a flight to city number {i}?"),
                    (
                        "ASSISTANT",
                        "I'm sorry, but I don't have the capability to book flights.",
                    ),
                ),
            }
        )
        rows.append(
            {
                "system": SYSTEM,
                "chat": chat(
                    ("USER", f"I want to convert some money, case {i}."),
                    (
                        "ASSISTANT",
                        "Sure, could you please provide the amount and the currencies?",
                    ),
                    ("USER", f"{200 + i} GBP to JPY"),
                    (
                        "ASSISTANT",
                        call(
                            "convert_currency",
                            {
                                "amount": 200 + i,
                                "from_currency": "GBP",
                                "to_currency": "JPY",
                            },
                        ),
                    ),
                ),
            }
        )
    for i in range(12):
        other = {
            "name": f"track_{['parcel', 'flight', 'fitness', 'sleep', 'mood', 'budget'][i % 6]}_{i}",
            "description": f"Track thing {i}",
            "parameters": {},
        }
        rows.append(
            {
                "system": "SYSTEM: You are a helpful assistant with access to the following functions. "
                "Use them if required -\n" + json.dumps(other),
                "chat": chat(
                    ("USER", f"Track item {i}"), ("ASSISTANT", call(other["name"], {}))
                ),
            }
        )
    return rows


class GlaiveFamiliesTest(unittest.TestCase):
    def test_family_rows_are_valid_and_labelled_by_rule(self):
        reports = collections.defaultdict(collections.Counter)
        convs = fam.glaive_conversations(conversations(), counter())
        values = collections.defaultdict(dict)
        for conv in convs:
            for name, args in conv["calls"]:
                for param, value in args.items():
                    values[(name, param)].setdefault(fam.dumps(value), value)
        pool = sorted({f["name"] for c in convs for f in c["functions"]})
        describe = {f["name"]: f["description"] for c in convs for f in c["functions"]}
        lists = [(c["key"], c["functions"]) for c in convs]
        rel = [r for c in convs for r in fam.fc_rel(c, reports["fc_rel"], lists=lists)]
        kinds = collections.Counter(
            (r["audit_metadata"]["ib2"]["kind"], r["label"]) for r in rel
        )
        self.assertEqual(kinds[("call", 1)], 52)
        self.assertEqual(kinds[("refusal", 0)], 40)
        self.assertGreater(kinds[("constructed", 0)], 0)
        for item in rel:
            if item["audit_metadata"]["ib2"]["kind"] == "constructed":
                names = [f["name"] for f in json.loads(item["state"]["functions"])]
                if item["state"]["request"].startswith("Convert"):
                    self.assertTrue(all(n.startswith("track_") for n in names))
                else:
                    self.assertFalse(any(n.startswith("track_") for n in names))
        self.assertFalse(
            fam.unrelated(
                {"name": "calculate_tip", "description": "Calculate the tip amount"},
                {
                    "name": "calculate_percentage",
                    "description": "Calculate a percentage",
                },
                "What is 15% of 200?",
            )
        )
        ready = [r for c in convs if (r := fam.fc_ready(c, reports["fc_ready"]))]
        labels = collections.Counter(r["label"] for r in ready)
        self.assertEqual(labels, {1: 40, 0: 40})
        args_rows = [
            r for c in convs if (r := fam.fc_args(c, reports["fc_args"], values=values))
        ]
        self.assertEqual(len(args_rows), 40)
        for item in args_rows:
            gold = json.loads(item["options"][item["label"]]["description"])
            self.assertEqual(gold["from_currency"], "USD")
            others = [
                json.loads(o["description"])
                for k, o in enumerate(item["options"])
                if k != item["label"]
            ]
            self.assertTrue(all(o != gold for o in others))
        sel = [
            r
            for c in convs
            if (r := fam.fc_sel(c, reports["fc_sel"], pool=pool, describe=describe))
        ]
        self.assertTrue(sel)
        for item in sel:
            gold = item["options"][item["label"]]["description"]
            self.assertTrue(gold.startswith(("convert_currency", "track_")))
            if gold.startswith("convert_currency"):
                # get_news is listed in that conversation, so it is never a distractor there.
                self.assertFalse(
                    any(
                        o["description"].startswith("get_news") for o in item["options"]
                    )
                )
        for item in rel + ready + args_rows + sel:
            validate_row(item, "train")


class OtherFamiliesTest(unittest.TestCase):
    def test_ytspam_argq_hover(self):
        spam = fam.ytspam(
            {
                "Psy": [
                    {
                        "COMMENT_ID": str(i),
                        "CONTENT": f"comment &amp; {i}\ufeff",
                        "CLASS": str(int(i < 4)),
                    }
                    for i in range(10)
                ]
            },
            counter(),
        )
        self.assertEqual(collections.Counter(r["label"] for r in spam), {0: 4, 1: 4})
        self.assertTrue(all("&amp;" not in r["state"]["comment"] for r in spam))
        records = [
            {
                "argument": f"argument {t} {i}",
                "topic": f"We should do thing {t}",
                "stance_WA": "1" if i % 3 else "-1",
                "stance_WA_conf": "1.0" if i != 5 else "0.8",
            }
            for t in range(3)
            for i in range(9)
        ]
        rows = fam.argq(records, counter())
        cells = collections.Counter((r["state"]["statement"], r["label"]) for r in rows)
        self.assertTrue(all(cells[(s, 0)] == cells[(s, 1)] for s, _ in cells))
        claims = [
            {
                "uid": f"u{i}",
                "claim": f"claim {i}",
                "supporting_facts": [["A", 0], ["B", 1], ["A", 2]],
                "label": "SUPPORTED" if i % 2 else "NOT_SUPPORTED",
                "num_hops": 2 if i % 7 else 4,
                "hpqa_id": f"h{i}",
            }
            for i in range(60)
        ]
        hov = fam.hover(claims, {"A": "Text A.", "B": "Text B."}.get, counter())
        labels = collections.Counter(r["label"] for r in hov)
        self.assertEqual(labels[1] * 2, labels[0] * 3)
        self.assertTrue(
            all(r["state"]["evidence"].startswith("A\nText A.") for r in hov)
        )

    def test_qasc_arc_gsm_cnli(self):
        q = fam.qasc(
            [
                {
                    "id": f"q{i}",
                    "question": f"What forms rain {i}?",
                    "choices": {
                        "label": list("ABCDEFGH"),
                        "text": [
                            "clouds",
                            "rocks",
                            "sand",
                            "fire",
                            "ice",
                            "wood",
                            "salt",
                            "iron",
                        ],
                    },
                    "answerKey": "A",
                    "combinedfact": (
                        "Rain is formed by clouds."
                        if i
                        else "Rain forms from ice and clouds."
                    ),
                }
                for i in range(5)
            ],
            counter(),
        )
        self.assertEqual(len(q), 4)
        self.assertTrue(
            all(r["options"][r["label"]]["description"] == "clouds" for r in q)
        )
        a = fam.arc(
            [
                (
                    "ARC-Easy",
                    {
                        "id": f"a{i}",
                        "question": f"Question {i}?",
                        "choices": {"label": ["1", "2", "3"], "text": ["x", "y", "z"]},
                        "answerKey": "2",
                    },
                )
                for i in range(6)
            ],
            counter(),
        )
        self.assertTrue(all(r["options"][r["label"]]["description"] == "y" for r in a))
        g = fam.gsm(
            [
                {
                    "question": f"Problem {i}?",
                    "answer": "48/2 = <<48/2=24>>24\n48+24 = <<48+24=72>>72\n#### 72",
                }
                for i in range(40)
            ],
            counter(),
        )
        self.assertEqual(
            collections.Counter(r["label"] for r in g)[0],
            collections.Counter(r["label"] for r in g)[1],
        )
        for item in g:
            shown = item["state"]["claim"]
            self.assertEqual(item["label"] == 1, shown == "The answer is 72.")
            self.assertIn(shown, ("The answer is 72.", "The answer is 24."))
        self.assertEqual(fam.number_text("1,000"), "1000")
        self.assertEqual(fam.number_text("2.50"), "2.5")
        data = {
            "labels": {k: {"hypothesis": v} for k, v in fam.CNLI_HYPOTHESES.items()},
            "documents": [
                {
                    "id": d,
                    "text": f"Contract {d}.",
                    "annotation_sets": [
                        {
                            "annotations": {
                                k: {
                                    "choice": (
                                        "Entailment",
                                        "NotMentioned",
                                        "Contradiction",
                                    )[(d + n) % 3]
                                }
                                for n, k in enumerate(sorted(fam.CNLI_HYPOTHESES))
                            }
                        }
                    ],
                }
                for d in range(9)
            ],
        }
        c = fam.cnli(data, counter())
        cells = collections.Counter(
            (r["audit_metadata"]["ib2"]["cell"], r["label"]) for r in c
        )
        self.assertTrue(all(cells[(h, 0)] == cells[(h, 1)] for h, _ in cells))
        self.assertTrue(all(r["instructions"] in fam.TEMPLATE_STRINGS for r in c))


class BalanceTest(unittest.TestCase):
    def test_rebalance_then_audit_passes(self):
        rows = fam.gsm2(
            [
                {
                    "question": f"Problem {i}?",
                    "answer": f"<<2*{i}={2 * i}>>{2 * i}\n#### {2 * i + 1}",
                }
                for i in range(1, 80)
            ],
            counter(),
        )
        golds = {f"Problem {i}?": f"The answer is {2 * i + 1}." for i in range(1, 80)}
        shown = collections.Counter(r["state"]["claim"] for r in rows)
        for item in rows:
            self.assertEqual(
                item["label"] == 1,
                golds[item["state"]["problem"]] == item["state"]["claim"],
            )
        self.assertTrue(set(shown) <= set(golds.values()))
        rows = build.rebalance(rows[3:])
        result, fails = audit.balance(rows)
        self.assertFalse(fails)
        self.assertTrue(result["gsm2"]["pass"])

    def test_redesigned_tool_families(self):
        convs = fam.glaive_conversations(conversations(), counter())
        called = [(c["key"], c["call1"][0]) for c in convs if c["call1"]]
        instances = collections.defaultdict(list)
        for conv in convs:
            for index, (name, args) in enumerate(conv["calls"]):
                for param, value in args.items():
                    instances[(name, param)].append((f"{conv['key']}:{index}", value))
        describe = {f["name"]: f["description"] for c in convs for f in c["functions"]}
        sel = [
            r
            for c in convs
            if (r := fam.fc_sel2(c, counter(), called=called, describe=describe))
        ]
        self.assertTrue(sel)
        for item in sel:
            texts = [o["description"] for o in item["options"]]
            self.assertEqual(len(set(texts)), 4)
        args_rows = [
            r for c in convs if (r := fam.fc_args2(c, counter(), instances=instances))
        ]
        self.assertEqual(len(args_rows), 40)
        for item in args_rows:
            gold = json.loads(item["options"][item["label"]]["description"])
            request = fam.Request(item["state"]["request"])
            for k, option in enumerate(item["options"]):
                if k != item["label"]:
                    other = json.loads(option["description"])
                    changed = [p for p in gold if other[p] != gold[p]]
                    self.assertEqual(len(changed), 1)
                    self.assertFalse(request.grounded(other[changed[0]]))
            validate_row(item, "train")


class ReviewSampleTest(unittest.TestCase):
    def test_review_sample_skips_screened_rows_and_stratifies(self):
        rows = fam.gsm2(
            [{"question": f"P{i}?", "answer": f"#### {i}"} for i in range(1, 400)],
            counter(),
        )
        screened = [{"id": r["id"], "group_id": r["group_id"]} for r in rows[:5]]
        built = review.review_sample(rows, "x", screened, set())
        self.assertEqual(built["sample"]["n"], 216)
        self.assertEqual(
            built["sample"]["cells"], {"gsm2|false": 108, "gsm2|true": 108}
        )
        self.assertFalse({k["id"] for k in built["key"]} & {k["id"] for k in screened})
        self.assertTrue(all(k["rid"].startswith("u") for k in built["key"]))


if __name__ == "__main__":
    unittest.main()
