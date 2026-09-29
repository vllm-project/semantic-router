import collections
import json
import unittest
from unittest import mock

from training.model.data import INPUT_FIELDS, digest, validate_row
from training.model.infer import load_prompts, question_to_row
from v2.data.m4 import pn1_build as build
from v2.data.m4 import pn1_source as source
from v2.data.m4 import pn1_text as text


def record(
    cid,
    lang,
    family,
    label,
    texts,
    *,
    group_key=None,
    ids=None,
    edited=None,
    kind=None,
    overlap_bin=None,
    coords=None,
):
    ids = ids or [int(part) for part in cid.split(":")[1:3] if part.isdigit()]
    edited = edited or (
        [False, True] if family in ("pn-name", "pn-twin") else [False, False]
    )
    item = source.candidate(
        None,
        family=family,
        lang=lang,
        cid=cid,
        group_key=group_key
        or (f"seed:{ids[0]}" if len(ids) == 1 else f"pair:{ids[0]}:{ids[1]}"),
        ids=ids,
        texts=texts,
        edited=edited,
        label=label,
        extra={"kind": kind, "editor": "rule" if family == "pn-name" else None},
    )
    if overlap_bin is not None:
        item["metrics"] = dict(item["metrics"], bin=overlap_bin)
    if coords is not None:
        item["metrics"] = dict(
            item["metrics"],
            bigram_jaccard=coords[0],
            multiset_jaccard=coords[1],
            length_ratio=coords[2],
            same_multiset=family in ("pn-name",),
        )
    item["attribution"] = {
        str(sid): {"user": f"user{sid}", "cc0": sid % 2 == 0} for sid in ids
    }
    item["judge"] = {
        "complete": True,
        "label": 0.9 if label else 0.1,
        "label_order": "ab",
        "fluency": 0.8 if any(edited) else None,
    }
    return build.with_strata([item])[0]


class NormalizationTest(unittest.TestCase):
    def test_norm_removes_space_and_punctuation_after_nfkc_casefold(self):
        self.assertEqual(text.norm("Tom's  HERE!"), "tomshere")
        self.assertEqual(text.norm("ＴＯＭ、来た。"), "tom来た")
        self.assertEqual(text.norm("「トム」は…来た？"), "トムは来た")
        self.assertEqual(text.norm("Straße"), "strasse")

    def test_usable_rejects_control_characters_and_empty_norms(self):
        self.assertTrue(text.usable("Hallo."))
        self.assertFalse(text.usable("a\tb"))
        self.assertFalse(text.usable("?!"))


class OverlapTest(unittest.TestCase):
    def test_bigram_and_multiset_jaccard(self):
        self.assertAlmostEqual(text.bigram_jaccard("abcd", "abce"), 2 / 4)
        self.assertAlmostEqual(text.multiset_jaccard("aab", "abb"), 2 / 4)
        self.assertEqual(text.bigram_jaccard("", ""), 1.0)
        self.assertTrue(text.same_multiset("abc", "cab"))
        self.assertFalse(text.same_multiset("aab", "ab"))

    def test_length_ratio_containment_and_bins(self):
        self.assertAlmostEqual(text.length_ratio("abcd", "ab"), 2.0)
        self.assertAlmostEqual(text.containment("abcde", "xabcdex"), 1.0)
        self.assertEqual(text.containment("abc", "abcd"), 0.0)
        self.assertTrue(text.near_duplicate("abcdefgh", "abcdefghij"))
        self.assertFalse(text.near_duplicate("abcdefgh", "abcdxyzw"))
        self.assertEqual(
            [text.overlap_bin(v) for v in (0.0, 0.3, 0.49, 0.5, 0.7, 0.85, 1.0)],
            [0, 1, 1, 2, 3, 4, 4],
        )

    def test_pair_metrics_of_a_swap(self):
        metrics = text.pair_metrics(
            "トムとメアリーは友達です。", "メアリーとトムは友達です。"
        )
        self.assertTrue(metrics["same_multiset"])
        self.assertEqual(metrics["multiset_jaccard"], 1.0)
        self.assertEqual(metrics["bin"], 2)
        self.assertEqual(text.stratum("pn-name", metrics), "swap|ms1|b2")


class KoreanAllomorphyTest(unittest.TestCase):
    def test_particles_follow_the_final_consonant(self):
        self.assertEqual(text.hangul_final("톰"), 16)
        self.assertEqual(text.hangul_final("리"), 0)
        for base, tom, mary in (
            ("이", "이", "가"),
            ("는", "은", "는"),
            ("를", "을", "를"),
            ("와", "과", "와"),
            ("랑", "이랑", "랑"),
        ):
            self.assertEqual(text.ko_particle_after(base, "톰"), tom)
            self.assertEqual(text.ko_particle_after(base, "메리"), mary)
        self.assertEqual(text.ko_particle_after("으로", "메리"), "로")
        self.assertEqual(text.ko_particle_after("로", "존"), "으로")
        self.assertEqual(text.ko_particle_after("으로", "서울"), "로")
        self.assertEqual(text.ko_particle_after("에게", "메리"), "에게")

    def test_particle_split(self):
        self.assertEqual(text.ko_split_particle("에게는"), ("에게", "는"))
        self.assertEqual(text.ko_split_particle("이랑"), ("이랑", ""))
        self.assertEqual(text.ko_split_particle("씨가"), ("씨", "가"))
        self.assertIsNone(text.ko_split_particle("재"))
        self.assertIsNone(text.ko_split_particle("이다"))
        self.assertIsNone(text.ko_split_particle("이는"))

    def test_korean_swaps_reselect_particles(self):
        self.assertEqual(
            text.name_swap("톰과 메리는 친구다.", "ko"),
            ("coord", "메리와 톰은 친구다."),
        )
        self.assertEqual(
            text.name_swap("톰이 메리를 좋아한다.", "ko"),
            ("role", "메리가 톰을 좋아한다."),
        )
        self.assertEqual(
            text.name_swap("메리는 존으로 착각했다.", "ko"),
            ("role", "존은 메리로 착각했다."),
        )
        self.assertEqual(
            text.name_swap("톰하고 메리가 왔다.", "ko"),
            ("coord", "메리하고 톰이 왔다."),
        )
        self.assertIsNone(text.name_swap("톰은 메리와 이야기했다.", "ko"))
        self.assertIsNone(text.name_swap("톰은 메리의 존재를 몰랐다.", "ko"))


class SwapTest(unittest.TestCase):
    CASES = {
        ("ja", "トムとメアリーは友達です。"): ("coord", "メアリーとトムは友達です。"),
        ("ja", "トムはメアリーを愛している。"): (
            "role",
            "メアリーはトムを愛している。",
        ),
        ("zh", "汤姆和玛丽是朋友。"): ("coord", "玛丽和汤姆是朋友。"),
        ("zh", "汤姆喜欢玛丽。"): ("role", "玛丽喜欢汤姆。"),
        ("es", "Tom y Mary son amigos."): ("coord", "Mary y Tom son amigos."),
        ("es", "Tom quiere a María."): ("role", "María quiere a Tom."),
        ("de", "Tom und Maria sind verheiratet."): (
            "coord",
            "Maria und Tom sind verheiratet.",
        ),
        ("de", "Tom hat Maria geküsst."): ("role", "Maria hat Tom geküsst."),
        ("fr", "Tom et Marie sont partis."): ("coord", "Marie et Tom sont partis."),
        ("fr", "Tom aime Marie."): ("role", "Marie aime Tom."),
        ("ru", "Том и Мэри друзья."): ("coord", "Мэри и Том друзья."),
        ("ru", "Том любит Мэри."): ("role", "Мэри любит Том."),
        ("ar", "توم وماري صديقان."): ("coord", "ماري وتوم صديقان."),
        ("ar", "توم يحب ماري."): ("role", "ماري يحب توم."),
    }
    REJECTED = (
        ("ja", "トムはメアリーと話した。"),
        ("ja", "トムソンはメアリーに会った。"),
        ("zh", "玛丽亚喜欢汤姆。"),
        ("es", "Tom habló con Mary."),
        ("de", "Tom ging mit Maria."),
        ("de", "Toms Frau Maria ist hier."),
        ("fr", "Marie-Claire aime Tom."),
        ("ru", "Том говорил с Мэри."),
        ("ar", "ذهب توم مع ماري."),
        ("de", "Tom sah Maria und John."),
        ("fr", "Tom aime Tom et Marie."),
        ("es", "Tom está aquí."),
    )

    def test_coordination_and_role_swaps(self):
        for (lang, sentence), expected in self.CASES.items():
            with self.subTest(lang=lang, sentence=sentence):
                self.assertEqual(text.name_swap(sentence, lang), expected)

    def test_rejections(self):
        for lang, sentence in self.REJECTED:
            with self.subTest(lang=lang, sentence=sentence):
                self.assertIsNone(text.name_swap(sentence, lang))

    def test_name_candidates_label_coordination_yes_and_role_no(self):
        found = source.name_candidates(
            "fr", {1: "Tom et Marie sont partis.", 2: "Tom aime Marie.", 3: "Bonjour."}
        )
        self.assertEqual(
            [(r["cid"], r["label"]) for r in found],
            [("name:1:coord", 1), ("name:2:role", 0)],
        )
        self.assertEqual(found[0]["edited"], [False, True])


class TwinTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(
            text.parse_twin('{"different": " a ", "same": "b"}'),
            {"different": "a", "same": "b"},
        )
        self.assertEqual(
            text.parse_twin('```json\n{"different": "a", "same": "b"}\n```'),
            {"different": "a", "same": "b"},
        )
        self.assertIsNone(text.parse_twin("Here: {}"))
        self.assertIsNone(text.parse_twin('{"different": "a"}'))
        self.assertIsNone(text.parse_twin('{"different": "a", "same": 3}'))

    def test_checks(self):
        seed = "Der Hund jagt die Katze durch den großen Garten."
        edits = {
            "different": "Die Katze jagt den Hund durch den großen Garten.",
            "same": seed,
        }
        reasons = text.twin_checks(seed, edits, "de")
        self.assertIsNone(reasons["different"])
        self.assertEqual(reasons["same"], "identical")
        edits = {
            "different": "Совсем другое предложение здесь.",
            "same": "Der Hund jagt durch den großen Garten die Katze.",
        }
        reasons = text.twin_checks(seed, edits, "de")
        self.assertIn(reasons["different"], ("multiset_jaccard", "script"))
        self.assertIsNone(reasons["same"])
        same = "Der Hund jagt durch den großen Garten die Katze."
        self.assertEqual(
            text.twin_checks(seed, {"different": same, "same": same}, "de"),
            {"different": "edits_identical", "same": "edits_identical"},
        )

    def test_script(self):
        self.assertTrue(text.script_ok("猫が好きです。", "犬が好きです。", "ja"))
        self.assertFalse(text.script_ok("猫が好き", "我喜欢猫", "zh"))
        self.assertTrue(text.script_ok("我喜欢狗", "我喜欢猫", "zh"))
        self.assertFalse(text.script_ok("Ich mag Hunde", "Я люблю собак", "ru"))

    def test_eligibility(self):
        self.assertTrue(
            text.twin_eligible("Tom est allé au marché avec sa sœur hier.", "fr")
        )
        self.assertFalse(text.twin_eligible("Tom est là.", "fr"))
        self.assertTrue(text.twin_eligible("나는 어제 친구와 함께 영화를 봤다.", "ko"))
        self.assertFalse(text.twin_eligible("短い文です。", "ja"))

    def test_prompts_name_the_language(self):
        self.assertIn("Japanese", text.twin_prompt("ja", "x"))
        self.assertTrue(
            text.label_prompt("ko", "a", "b").startswith(
                "Here are two sentences in Korean.\nSentence A: a\nSentence B: b\n"
            )
        )
        self.assertEqual(
            text.fluency_prompt("de", "s"),
            "Is the following sentence natural and grammatical German?\ns\nAnswer Yes or No.",
        )


class PromptV2Test(unittest.TestCase):
    def test_v2_prompts_and_extra_metrics(self):
        label = text.label_prompt_v2("ja", "a", "b")
        self.assertTrue(
            label.startswith(
                "Here are two sentences in Japanese.\nSentence A: a\nSentence B: b\nDo the two sentences mean the same thing?"
            )
        )
        self.assertTrue(label.endswith("or in any other fact. Answer Yes or No."))
        fluency = text.fluency_prompt_v2("de", "s")
        self.assertTrue(
            fluency.startswith(
                "Here is a sentence in German:\ns\nIs this a grammatical sentence"
            )
        )
        self.assertAlmostEqual(text.edit_distance("abcd", "abce"), 0.25)
        self.assertEqual(text.edit_distance("", ""), 0.0)
        extra = text.extra_metrics("Ich esse einen Apfel.", "Ich esse eine Birne.")
        self.assertAlmostEqual(extra["word_jaccard"], 2 / 6, places=5)
        self.assertEqual(
            set(extra),
            {"word_jaccard", "edit_distance", "containment_ab", "containment_ba"},
        )


class GraphTest(unittest.TestCase):
    def test_within_two_hops(self):
        graph = source.LinkGraph.from_pairs([(1, 2), (2, 3), (3, 4), (5, 6)])
        self.assertEqual(graph.within([1]), {1, 2, 3})
        self.assertEqual(graph.within([1], hops=1), {1, 2})
        self.assertEqual(graph.within([4, 5]), {2, 3, 4, 5, 6})
        self.assertEqual(graph.edges, 4)

    def test_hop_pairs_one_per_pivot(self):
        texts = {
            1: "Je suis très fatigué ce soir.",
            2: "Je suis très fatiguée ce soir.",
            3: "Je suis crevé ce soir.",
            4: "Il pleut.",
        }
        graph = source.LinkGraph.from_pairs([(1, 100), (2, 100), (3, 100), (4, 101)])
        lang_of = {**{sid: "fr" for sid in texts}, 100: "en", 101: "en"}
        pairs, stats = source.hop_pairs("fr", texts, graph, lang_of, set(), 10)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["pivot"], 100)
        self.assertEqual(pairs[0]["label"], 1)
        self.assertGreaterEqual(pairs[0]["metrics"]["bin"], 1)
        pairs, _ = source.hop_pairs("fr", texts, graph, lang_of, {1, 2}, 10)
        self.assertEqual(pairs, [])

    def test_near_pairs_respect_links_and_quotas(self):
        texts = {
            1: "Je mange une pomme rouge.",
            2: "Je mange une pomme verte.",
            3: "Tu manges une pomme rouge.",
            9: "Rien à voir ici du tout.",
        }
        graph = source.LinkGraph.from_pairs([(1, 50), (2, 50)])
        quotas = collections.Counter()
        edges = []
        for a, b in ((1, 2), (1, 3), (2, 3)):
            m = text.pair_metrics(texts[a], texts[b])
            na, nb = text.norm(texts[a]), text.norm(texts[b])
            quotas[
                source.fine_cell(
                    m["bigram_jaccard"],
                    m["length_ratio"],
                    (len(na) + len(nb)) / 2,
                    edges,
                )
            ] += 1
        pairs, stats = source.near_pairs(
            "fr", texts, graph, set(), lambda sid: set(), dict(quotas), edges
        )
        self.assertEqual(len(pairs), 1)
        self.assertNotEqual(set(pairs[0]["ids"]), {1, 2})
        self.assertEqual(pairs[0]["label"], 0)
        self.assertGreaterEqual(pairs[0]["metrics"]["bigram_jaccard"], 0.3)


class BalanceTest(unittest.TestCase):
    def test_allocate_is_proportional_capped_and_exact(self):
        self.assertEqual(
            build.allocate(10, {"a": 1, "b": 1}, {"a": 100, "b": 100}), {"a": 5, "b": 5}
        )
        self.assertEqual(
            build.allocate(10, {"a": 3, "b": 1}, {"a": 2, "b": 100}), {"a": 2, "b": 8}
        )
        self.assertEqual(
            sum(
                build.allocate(
                    7, {"a": 1, "b": 1, "c": 1}, {"a": 9, "b": 9, "c": 9}
                ).values()
            ),
            7,
        )
        self.assertEqual(build.allocate(10, {"a": 1}, {"a": 4}), {"a": 4})

    def test_group_targets(self):
        self.assertEqual(
            build.group_targets("ja", 4400), {"natural": 2200, "swap": 2200}
        )
        self.assertEqual(
            build.group_targets("de", 2000), {"natural": 1200, "swap": 800}
        )
        self.assertEqual(build.group_targets("ru", 100), {"natural": 60, "swap": 40})

    def pool(self):
        rows = []
        for i in range(40):
            b = 1 + i % 3
            rows.append(
                record(
                    f"hop:{1000 + 2 * i}:{1001 + 2 * i}",
                    "de",
                    "pn-hop",
                    1,
                    [f"Das ist Satz {i} hier.", f"Das ist der Satz {i} hier."],
                    overlap_bin=b,
                    coords=(0.3 + 0.01 * i, 0.7, 1.1),
                )
            )
            rows.append(
                record(
                    f"near:{2000 + 2 * i}:{2001 + 2 * i}",
                    "de",
                    "pn-near",
                    0,
                    [f"Er kam um {i} Uhr.", f"Sie kam um {i} Uhr."],
                    overlap_bin=b,
                    coords=(0.305 + 0.01 * i, 0.71, 1.12),
                )
            )
        for i in range(30):
            rows.append(
                record(
                    f"name:{3000 + i}:coord",
                    "de",
                    "pn-name",
                    1,
                    [f"Tom und Maria sind {i} hier.", f"Maria und Tom sind {i} hier."],
                    kind="coord",
                    overlap_bin=3,
                    coords=(0.7 + 0.005 * i, 1.0, 1.0),
                )
            )
            rows.append(
                record(
                    f"name:{4000 + i}:role",
                    "de",
                    "pn-name",
                    0,
                    [f"Tom sah Maria {i} mal.", f"Maria sah Tom {i} mal."],
                    kind="role",
                    overlap_bin=3,
                    coords=(0.7 + 0.005 * i, 1.0, 1.0),
                )
            )
        return rows

    def test_balance_equalizes_every_stratum(self):
        chosen, units = build.balance(self.pool(), {"natural": 30, "swap": 20})
        cells = collections.Counter((r["stratum"], r["label"]) for r in chosen)
        for s in {s for s, _ in cells}:
            self.assertEqual(cells[(s, 0)], cells[(s, 1)])
        self.assertEqual(sum(r["group"] == "natural" for r in chosen), 30)
        self.assertEqual(sum(r["group"] == "swap" for r in chosen), 20)
        again, _ = build.balance(
            list(reversed(self.pool())), {"natural": 30, "swap": 20}
        )
        self.assertEqual(
            sorted(r["cid"] for r in chosen), sorted(r["cid"] for r in again)
        )

    def test_matching_respects_the_caliper_and_order(self):
        pairs = build.match_pairs(self.pool())
        self.assertEqual(set(pairs), {"natural|ms0", "swap|ms1"})
        for stratum_pairs in pairs.values():
            nos = [no for no, _, _ in stratum_pairs]
            self.assertEqual(nos, sorted(nos, key=build.seed_key))
            for no, yes, _ in stratum_pairs:
                self.assertEqual((no["label"], yes["label"]), (0, 1))
                for c in build.MATCH_COORDINATES:
                    self.assertLessEqual(
                        abs(no["metrics"][c] - yes["metrics"][c]), build.CALIPER + 1e-9
                    )
        natural = len(pairs["natural|ms0"])
        self.assertGreaterEqual(natural, 36)
        far = record(
            "near:9000:9001",
            "de",
            "pn-near",
            0,
            ["a b", "c d"],
            coords=(0.95, 0.2, 1.5),
        )
        again = build.match_pairs(self.pool() + [far])["natural|ms0"]
        self.assertEqual(len(again), natural)
        self.assertNotIn("near:9000:9001", {no["cid"] for no, _, _ in again})

    def test_drops_rematch_a_balanced_subset_and_empty_drop_reproduces(self):
        chosen, _, _ = build.balance_matched(self.pool(), {"natural": 30, "swap": 20})
        self.assertEqual(len(chosen), 50)
        same = {"de": {"train": list(chosen), "dev": []}}
        build.apply_drops(same, set())
        self.assertEqual(
            {r["cid"] for r in same["de"]["train"]}, {r["cid"] for r in chosen}
        )
        dropped = {
            build.group_id(r["group_key"]) for r in chosen if r["family"] == "pn-hop"
        }
        selection = {"de": {"train": list(chosen), "dev": []}}
        report = build.apply_drops(selection, set(list(sorted(dropped))[:3]))
        left = selection["de"]["train"]
        self.assertEqual(report["de"]["train"]["dropped"], 3)
        self.assertTrue({r["cid"] for r in left} <= {r["cid"] for r in chosen})
        cells = collections.Counter((build.match_stratum(r), r["label"]) for r in left)
        for s in {s for s, _ in cells}:
            self.assertEqual(cells[(s, 0)], cells[(s, 1)])
        self.assertEqual(2 * sum(r["label"] for r in left), len(left))


class DevIsolationTest(unittest.TestCase):
    def test_blocklist_neighbourhood_norm_and_containment(self):
        graph = source.LinkGraph.from_pairs([(1, 90), (90, 91), (91, 5)])
        dev = [
            record(
                "hop:1:2",
                "de",
                "pn-hop",
                1,
                ["Ich gehe heute nach Hause.", "Heute gehe ich nach Hause."],
            )
        ]
        blocklist = build.Blocklist(dev, graph)
        self.assertIn(91, blocklist.neighbourhood)
        self.assertNotIn(5, blocklist.neighbourhood)
        self.assertEqual(
            blocklist.reason(
                record("near:91:7", "de", "pn-near", 0, ["x y z", "a b c"])
            ),
            "dev_neighbourhood",
        )
        self.assertEqual(
            blocklist.reason(
                record(
                    "near:6:7",
                    "de",
                    "pn-near",
                    0,
                    ["ICH gehe heute nach Hause!", "Etwas anderes."],
                )
            ),
            "dev_norm_equal",
        )
        self.assertEqual(
            blocklist.reason(
                record(
                    "near:8:9",
                    "de",
                    "pn-near",
                    0,
                    ["Ich gehe heute nach Hause, ja.", "Etwas anderes."],
                )
            ),
            "dev_near_duplicate",
        )
        self.assertIsNone(
            blocklist.reason(
                record(
                    "near:5:7",
                    "de",
                    "pn-near",
                    0,
                    ["Ganz andere Worte.", "Nichts davon."],
                )
            )
        )

    def test_select_language_dev_first_disjoint_and_balanced(self):
        pool = BalanceTest().pool()
        graph = source.LinkGraph.from_pairs([])
        with mock.patch.dict(build.TRAIN_TARGETS, {"de": 60}), mock.patch.dict(
            build.DEV_QUOTAS, {"de": 20}
        ):
            chosen = build.select_language(pool, "de", graph)
        dev, train = chosen["dev"], chosen["train"]
        self.assertEqual(len(dev), 20)
        self.assertEqual(2 * sum(r["label"] for r in dev), len(dev))
        self.assertEqual(2 * sum(r["label"] for r in train), len(train))
        self.assertFalse(
            {r["group_key"] for r in dev} & {r["group_key"] for r in train}
        )
        dev_ids = {sid for r in dev for sid in r["ids"]}
        self.assertFalse(dev_ids & {sid for r in train for sid in r["ids"]})
        self.assertTrue(chosen["mix"]["ok"])
        blocklist = build.Blocklist(dev, graph)
        self.assertTrue(all(blocklist.reason(r) is None for r in train))


class RowTest(unittest.TestCase):
    def test_rows_validate_and_render_from_the_row_id(self):
        item = record(
            "twin:77:different",
            "ja",
            "pn-twin",
            0,
            [
                "犬が猫を追いかけている公園の中で。",
                "猫が犬を追いかけている公園の中で。",
            ],
            kind="different",
        )
        item["editor"] = "qwen3.5-27b@rev"
        row = build.make_row(item, "train", "2026-09-26", "Qwen/Qwen3.8-27B@rev")
        self.assertTrue(row["id"].startswith("m4pn1-pn-twin-"))
        self.assertEqual(len(row["id"].rsplit("-", 1)[1]), 24)
        self.assertRegex(row["group_id"], r"^m4pn1:tatoeba:[0-9a-f]{24}$")
        self.assertEqual(row["source"], "tatoeba-2026-09-26-edited")
        self.assertEqual(
            row["options"],
            [
                {"key": "false", "description": "No"},
                {"key": "true", "description": "Yes"},
            ],
        )
        self.assertRegex(row["render_template"], r"^pn1/pn-twin/i[1-8]/l[1-5]$")
        self.assertNotIn("Sentence 1:", row["state"])
        self.assertEqual(row["audit_metadata"]["source_local_id"], "twin:77:different")
        validate_row(row, "train")
        dev = build.make_row(item, "dev", "2026-09-26", "Qwen/Qwen3.8-27B@rev")
        self.assertEqual((dev["split"], dev["evaluation_role"]), ("select", "select"))
        self.assertEqual(dev["id"], row["id"])
        natural = build.make_row(
            record("hop:5:6", "ja", "pn-hop", 1, ["a b", "c d"]),
            "train",
            "2026-09-26",
            "j",
        )
        self.assertEqual(natural["source"], "tatoeba-2026-09-26")

    def test_dev_prompt_round_trips_and_carries_no_gold(self):
        item = record(
            "near:5:6",
            "de",
            "pn-near",
            0,
            ["Ich esse einen Apfel.", "Ich esse eine Birne."],
        )
        row = build.make_row(item, "dev", "2026-09-26", "judge")
        prompt, gold = build.dev_prompt(row)
        self.assertEqual(set(prompt), {"id", "state", "questions"})
        question = prompt["questions"]["decision"]
        restored = question_to_row(prompt, "decision", question)
        self.assertEqual(
            digest({f: restored[f] for f in INPUT_FIELDS}), row["input_sha256"]
        )
        self.assertNotIn("label", json.dumps(prompt))
        self.assertIs(gold["gold"]["decision"]["value"], False)
        self.assertEqual(gold["source_item_id"], row["id"])

    def test_prompts_load_with_the_formal_reader(self):
        import tempfile
        from pathlib import Path

        rows = [
            build.make_row(
                record(
                    f"near:{i}:{i + 1}",
                    "fr",
                    "pn-near",
                    0,
                    [f"Phrase {i}.", f"Autre {i}."],
                ),
                "dev",
                "2026-09-26",
                "j",
            )
            for i in (10, 20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(
                "".join(
                    json.dumps(build.dev_prompt(r)[0], ensure_ascii=False) + "\n"
                    for r in rows
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(load_prompts(path)), 2)

    def test_instruction_layout_and_order_depend_only_on_the_row_id(self):
        seen = collections.Counter()
        for i in range(400):
            rid = f"m4pn1-pn-hop-{i:024d}"
            seen[
                (build.pick(rid, "instructions", 8), build.pick(rid, "layout", 5))
            ] += 1
        self.assertEqual(len(seen), 40)

    def test_attribution(self):
        items = [
            record(
                "name:4:coord",
                "fr",
                "pn-name",
                1,
                ["Tom et Marie.", "Marie et Tom."],
                kind="coord",
            ),
            record("hop:3:5", "fr", "pn-hop", 1, ["a b", "c d"]),
        ]
        lines = build.attribution_lines(items)
        self.assertEqual(lines[0], "sentence_id\tlanguage\tusername\tlicence\tedited")
        self.assertEqual(
            lines[1:],
            [
                "3\tfr\tuser3\tCC-BY-2.0-FR\t0",
                "4\tfr\tuser4\tCC0\t1",
                "5\tfr\tuser5\tCC-BY-2.0-FR\t0",
            ],
        )


class SelfCheckTest(unittest.TestCase):
    def test_logistic_regression_detects_a_feature_shortcut(self):
        rows = []
        for i in range(60):
            for label in (0, 1):
                rows.append(
                    {
                        "label": label,
                        "group_id": f"g{i}-{label}",
                        "audit_metadata": {
                            "overlap": {
                                "bigram_jaccard": 0.4 + 0.4 * label + (i % 7) / 100,
                                "multiset_jaccard": 0.8,
                                "same_multiset": False,
                                "length_ratio": 1.1,
                            }
                        },
                    }
                )
        result = build.cv_accuracy(rows)
        self.assertGreater(result["accuracy"], 0.95)
        self.assertLess(result["majority"], 0.6)


class PipelineTest(unittest.TestCase):
    """candidates -> (faked generations) -> judgeset -> (faked judgments) -> finalize -> drop re-run."""

    OBJECTS = (
        "une pomme",
        "un livre",
        "une lettre",
        "un gâteau",
        "une photo",
        "un vélo",
        "une clé",
        "un stylo",
        "une robe",
        "un chien",
        "une table",
        "un verre",
    )

    def export(self, root):
        import bz2
        import hashlib
        import io
        import tarfile

        fr, en, links, sid = [], [], [], 100
        for i, obj in enumerate(self.OBJECTS):
            pivot = 10_000 + i
            en.append((pivot, f"I see {obj} {i}.", "e"))
            a, b = sid, sid + 1
            fr += [
                (a, f"Je vois {obj} ici {i}.", f"u{a}"),
                (b, f"Je vois {obj} là {i}.", f"u{b}"),
            ]
            links += [(a, pivot), (pivot, a), (b, pivot), (pivot, b)]
            fr += [
                (sid + 2, f"Tu vois {obj} ici {i}.", f"u{sid + 2}"),
                (sid + 3, f"Tu vois {obj} là {i}.", f"u{sid + 3}"),
            ]
            fr += [
                (sid + 4, f"Tom et Marie voient {obj}.", "n"),
                (sid + 5, f"Tom voit Marie et {obj}.", "n"),
            ]
            fr.append(
                (
                    sid + 6,
                    f"Le petit chat noir mange {obj} dans la grande cuisine ce soir.",
                    "t",
                )
            )
            sid += 10
        tables = {"fr": fr, "en": en}
        files = source.export_files()
        for lang, iso3 in source.EXPORT_LANGUAGES.items():
            path = root / files[f"{iso3}_sentences_detailed"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with bz2.open(path, "wt", encoding="utf-8") as stream:
                for number, sentence, user in tables.get(lang, []):
                    stream.write(
                        f"{number}\t{iso3}\t{sentence}\t{user}\t2020-01-01\t2020-01-01\n"
                    )
        for name, member, body in (
            ("links", "links.csv", "".join(f"{a}\t{b}\n" for a, b in links)),
            ("sentences_CC0", "sentences_CC0.csv", "100\tfra\tx\t2020\n"),
        ):
            data = body.encode()
            with tarfile.open(root / files[name], "w:bz2") as archive:
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        manifest = {
            "export_date": "2026-09-26",
            "files": {
                rel: {"sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
                for rel in files.values()
            },
        }
        (root / "source-manifest.json").write_text(json.dumps(manifest))

    def fake_gpu(self, directory, name, records, model):
        directory.mkdir(parents=True)
        path = directory / name
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
        receipt = {
            "model": model,
            "outputs_sha256": build.file_sha256(path),
            "counts": {"n": len(records)},
            "timing": {},
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))

    def test_end_to_end(self):
        import tempfile
        from pathlib import Path

        targets = {lang: 0 for lang in text.LANGS}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            build.TRAIN_TARGETS, dict(targets, fr=16)
        ), mock.patch.dict(build.DEV_QUOTAS, dict(targets, fr=8)):
            tmp = Path(tmp)
            src, work, out = tmp / "src", tmp / "work", tmp / "out"
            src.mkdir()
            self.export(src)
            self.assertEqual(
                build.main(
                    [
                        "candidates",
                        "--source-dir",
                        str(src),
                        "--work-dir",
                        str(work),
                        "--workers",
                        "2",
                    ]
                ),
                0,
            )
            candidates = build.read_jsonl(work / "candidates.jsonl")
            families = collections.Counter(r["family"] for r in candidates)
            self.assertGreater(families["pn-hop"], 0)
            self.assertGreater(families["pn-name"], 0)
            seeds = build.read_jsonl(work / "seeds.jsonl")
            self.assertEqual(len(seeds), len(self.OBJECTS))
            generations = []
            for seed in seeds:
                words = seed["text"].split()
                different = words[:]
                different[2], different[3] = different[3], different[2]
                same = words[:]
                same[-2], same[-3] = same[-3], same[-2]
                generations.append(
                    {
                        "seed": seed["seed"],
                        "language": "fr",
                        "raw": json.dumps(
                            {"different": " ".join(different), "same": " ".join(same)}
                        ),
                        "new_tokens": 40,
                        "finished": True,
                    }
                )
            generator = {"repo_id": "Qwen/Qwen3.5-27B", "revision": "genrev"}
            self.fake_gpu(
                work / "gen" / "gen-gpu3", "generations.jsonl", generations, generator
            )
            self.assertEqual(
                build.main(
                    [
                        "judgeset",
                        "--work-dir",
                        str(work),
                        "--gen-dir",
                        str(work / "gen" / "gen-gpu3"),
                    ]
                ),
                0,
            )
            pool = {
                r["cid"]: r for r in build.read_jsonl(work / "judgeset" / "pool.jsonl")
            }
            self.assertTrue(any(r["family"] == "pn-twin" for r in pool.values()))
            judge_dirs = []
            for items_path in sorted((work / "judgeset").glob("items.*.jsonl")):
                judged = []
                for item in build.read_jsonl(items_path):
                    label = pool[item["cid"]]["label"]
                    p = (0.9 if label else 0.1) if item["kind"] == "label" else 0.9
                    judged.append(
                        {
                            "item": item["item"],
                            "cid": item["cid"],
                            "kind": item["kind"],
                            "language": "fr",
                            "order": item.get("order"),
                            "p_yes": p,
                        }
                    )
                directory = work / "judge" / items_path.stem.split(".")[1]
                self.fake_gpu(
                    directory,
                    "judgments.jsonl",
                    judged,
                    {"repo_id": "Qwen/Qwen3.8-27B", "revision": "judgerev"},
                )
                judge_dirs += ["--judge-dir", str(directory)]
            self.assertEqual(
                build.main(
                    [
                        "finalize",
                        "--work-dir",
                        str(work),
                        "--source-dir",
                        str(src),
                        "--output-root",
                        str(out),
                        *judge_dirs,
                    ]
                ),
                0,
            )
            (first,) = list(out.iterdir())
            manifest = json.loads((first / "build-manifest.json").read_text())
            train = [
                validate_row(json.loads(line), "train")
                for line in (first / "pn1.train.jsonl").read_text().splitlines()
            ]
            dev = [
                validate_row(json.loads(line), "select")
                for line in (first / "pn1.dev.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(dev), 8)
            self.assertGreater(len(train), 0)
            self.assertEqual(2 * sum(r["label"] for r in train), len(train))
            self.assertFalse(
                {r["group_id"] for r in train} & {r["group_id"] for r in dev}
            )
            prompts = load_prompts(first / "pn1.dev.prompts.jsonl")
            self.assertEqual(len(prompts), len(dev))
            gold = build.read_jsonl(first / "pn1.dev.gold.jsonl")
            self.assertEqual({g["id"] for g in gold}, {p["id"] for p in prompts})
            self.assertEqual(
                manifest["files_sha256"]["pn1.train.jsonl"],
                build.file_sha256(first / "pn1.train.jsonl"),
            )
            self.assertTrue(
                (first / "attribution.tsv")
                .read_text()
                .startswith("sentence_id\tlanguage\tusername\tlicence\tedited\n")
            )
            editors = {
                r["audit_metadata"]["editor"]
                for r in train + dev
                if r["family"] == "pn-twin"
            }
            self.assertLessEqual(editors, {"qwen3.5-27b@genrev"})

            drop = tmp / "drop.txt"
            victim = train[0]["group_id"]
            drop.write_text(f"# audit hit\n{victim}\n")
            self.assertEqual(
                build.main(
                    [
                        "finalize",
                        "--work-dir",
                        str(work),
                        "--source-dir",
                        str(src),
                        "--output-root",
                        str(out),
                        *judge_dirs,
                        "--drop-groups",
                        str(drop),
                        "--expect-selection",
                        manifest["selection_sha256_pre_drop"],
                    ]
                ),
                0,
            )
            (second,) = [d for d in out.iterdir() if d != first]
            again = [
                json.loads(line)
                for line in (second / "pn1.train.jsonl").read_text().splitlines()
            ]
            self.assertTrue({r["id"] for r in again} < {r["id"] for r in train})
            self.assertNotIn(victim, {r["group_id"] for r in again})
            self.assertEqual(2 * sum(r["label"] for r in again), len(again))
            with self.assertRaises(ValueError):
                build.main(
                    [
                        "finalize",
                        "--work-dir",
                        str(work),
                        "--source-dir",
                        str(src),
                        "--output-root",
                        str(out),
                        *judge_dirs,
                        "--expect-selection",
                        "0" * 64,
                    ]
                )

            extend = ["judgeset-extend", "--work-dir", str(work), *judge_dirs]
            self.assertEqual(build.main([*extend, "--dry-run"]), 0)
            self.assertEqual(build.main(extend), 0)
            extended = json.loads(
                (work / "judgeset2" / "judgeset.receipt.json").read_text()
            )
            self.assertEqual(extended["twin_status_file"], "judgeset/twin-status.jsonl")
            fin = ["finalize", "--work-dir", str(work), "--source-dir", str(src)]
            fin += [
                "--output-root",
                str(tmp / "out2"),
                *judge_dirs,
                "--judgeset",
                "judgeset2",
            ]
            self.assertEqual(build.main(fin), 0)
            (third,) = list((tmp / "out2").iterdir())
            manifest3 = json.loads((third / "build-manifest.json").read_text())
            self.assertIsNotNone(manifest3["steps"]["judge_extension"])

            self.assertEqual(build.main(["probe-items", "--work-dir", str(work)]), 0)
            probe = build.read_jsonl(work / "probe" / "items.probe.jsonl")
            self.assertEqual({i["item"].split(":")[1] for i in probe}, {"v1", "v2"})
            answers = [dict(i, p_yes=0.99) for i in probe]
            self.fake_gpu(
                work / "judge" / "probe",
                "judgments.jsonl",
                answers,
                {"repo_id": "Qwen/Qwen3.8-27B", "revision": "judgerev"},
            )
            code = build.main(
                [
                    "probe-eval",
                    "--judge-dir",
                    str(work / "judge" / "probe"),
                    "--output",
                    str(work / "probe" / "probe.receipt.json"),
                ]
            )
            self.assertEqual(code, 4)
            self.assertFalse(
                json.loads((work / "probe" / "probe.receipt.json").read_text())[
                    "passed"
                ]
            )
            self.assertEqual(
                build.main(["judgeset-rejudge", "--work-dir", str(work)]), 0
            )
            rejudge = [
                i
                for p in sorted((work / "judgeset3").glob("items.*.jsonl"))
                for i in build.read_jsonl(p)
            ]
            judged3 = {i["cid"] for i in rejudge}
            swap = {c for c, r in pool.items() if r["family"] in ("pn-name", "pn-twin")}
            self.assertTrue(swap <= judged3)
            pool3 = {
                r["cid"]: r for r in build.read_jsonl(work / "judgeset3" / "pool.jsonl")
            }
            for cid in judged3 - swap:
                self.assertLess(pool3[cid]["rank"], 1.3)
            order = [pool3[i["cid"]]["tier"] for i in rejudge if i["kind"] == "label"]
            self.assertEqual(order, sorted(order))
            self.assertTrue(
                all(
                    "do not matter" in i["prompt"] or "native speaker" in i["prompt"]
                    for i in rejudge
                )
            )
            receipt3 = json.loads(
                (work / "judgeset3" / "judgeset.receipt.json").read_text()
            )
            self.assertEqual(receipt3["prompt_version"], "v2")


class JudgeExtensionTest(unittest.TestCase):
    def test_extension_adds_rows_where_measured_keeps_fall_short(self):
        pool = []
        for i in range(40):
            hop = record(
                f"hop:{100 + 2 * i}:{101 + 2 * i}",
                "de",
                "pn-hop",
                1,
                [f"Er geht {i} nach Hause.", f"Er läuft {i} heim."],
                overlap_bin=1,
            )
            near = record(
                f"near:{300 + 2 * i}:{301 + 2 * i}",
                "de",
                "pn-near",
                0,
                [f"Sie kommt {i} spät.", f"Wir essen {i} Brot."],
                overlap_bin=1,
            )
            for item in (hop, near):
                item["stratum"] = "natural|ms0|b1"
                item["judged"] = False
            pool += [hop, near]
        ordered_hops = sorted(
            (r for r in pool if r["family"] == "pn-hop"), key=build.seed_key
        )
        ordered_nears = sorted(
            (r for r in pool if r["family"] == "pn-near"), key=build.seed_key
        )
        outcome = {r["cid"]: "not_judged" for r in pool}
        for index, item in enumerate(ordered_hops[:10]):
            item["judged"] = True
            outcome[item["cid"]] = "kept" if index % 2 else "rejected_label"
        for item in ordered_nears[:10]:
            item["judged"] = True
            outcome[item["cid"]] = "kept"
        zero = {lang: 0 for lang in text.LANGS}
        with mock.patch.dict(build.TRAIN_TARGETS, dict(zero, de=20)), mock.patch.dict(
            build.DEV_QUOTAS, zero
        ):
            added, plan = build.extension(pool, outcome, 1.25)
        self.assertEqual(plan["keep_rates"]["de|pn-hop|1"], 0.5)
        self.assertEqual(
            [r["cid"] for r in added], [r["cid"] for r in ordered_hops[10:15]]
        )
        cell = plan["cells"]["de|natural|ms0|b1|y1"]
        self.assertEqual(
            (cell["planned_units"], cell["kept_pass1"], cell["added"]), (6, 5, 5)
        )
        self.assertEqual(plan["cells"]["de|natural|ms0|b1|y0"]["added"], 0)


class GpuCostTest(unittest.TestCase):
    def test_cost_uses_repeated_widths_and_charges_new_ones(self):
        from v2.data.m4 import pn1_gpu

        runner = pn1_gpu.Runner(budget=100.0, size=4)
        runner.log = [
            {
                "phase": "preflight",
                "items": 4,
                "width": 64,
                "seconds": 20.0,
                "new_shape": True,
            },
            {
                "phase": "preflight",
                "items": 4,
                "width": 64,
                "seconds": 5.0,
                "new_shape": False,
            },
            {
                "phase": "preflight",
                "items": 1,
                "width": 96,
                "seconds": 25.0,
                "new_shape": True,
            },
        ]
        runner.shapes = {64, 96}
        self.assertEqual(runner.batch_seconds(), 5.0)
        known = [[{"ids": [1] * 60}]] * 3
        self.assertAlmostEqual(runner.cost(known), pn1_gpu.SAFETY * 3 * 5.0)
        fresh = known + [[{"ids": [1] * 120}]]
        self.assertAlmostEqual(
            runner.cost(fresh), pn1_gpu.SAFETY * 4 * 5.0 + pn1_gpu.SHAPE_ALLOWANCE
        )
        args = pn1_gpu.main.__globals__["argparse"].Namespace()
        self.assertIsNotNone(args)
        from unittest import mock as _mock

        with _mock.patch.object(
            pn1_gpu, "cmd_judge", lambda a: a.shape_allowance
        ), _mock.patch.object(pn1_gpu, "cmd_generate", lambda a: a.shape_allowance):
            common = [
                "--model-dir",
                "m",
                "--repo-id",
                "r",
                "--revision",
                "v",
                "--output-dir",
                "o",
                "--budget-seconds",
                "5",
            ]
            self.assertEqual(
                pn1_gpu.main(
                    [
                        "judge",
                        *common,
                        "--items",
                        "i",
                        "--shape-allowance",
                        "3",
                        "--min-free-gb",
                        "100",
                    ]
                ),
                3.0,
            )
            self.assertEqual(
                pn1_gpu.main(
                    ["generate", *common, "--seeds", "s", "--languages", "ja"]
                ),
                pn1_gpu.SHAPE_ALLOWANCE,
            )
        runner.log = runner.log[:1] + runner.log[2:]
        self.assertEqual(runner.batch_seconds(), 25.0)


class RegistryTest(unittest.TestCase):
    def test_registry_has_both_keys(self):
        from pathlib import Path

        path = (
            Path(build.__file__).resolve().parents[1]
            / "records"
            / "license-registry-m4.json"
        )
        registry = json.loads(path.read_text(encoding="utf-8"))
        for key in ("tatoeba-2026-09-26", "tatoeba-2026-09-26-edited"):
            self.assertTrue(
                {"license", "attribution", "evidence", "redistribution"}
                <= set(registry[key])
            )
            self.assertTrue(registry[key]["license"].startswith("CC BY 2.0 FR"))


if __name__ == "__main__":
    unittest.main()
