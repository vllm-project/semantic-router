"""Unit tests for the audit_lib modules: parsing, storage, statistics, review and report."""

import json
import random
import types

import pytest
from audit_lib import api_judge, human_review, report
from audit_lib.checkpoint import (
    append_records,
    count_judged,
    read_checkpoint,
    records_for_split,
    select_ids,
)
from audit_lib.dataset import (
    DatasetMismatchError,
    clip,
    load_rows,
    rubric_hash,
    verify_pinned,
)
from audit_lib.judgment import (
    Judgment,
    build_user_message,
    inclusive_label,
    make_record,
    parse_line,
    parse_lines,
    save_judgments,
)
from audit_lib.stats import (
    accuracy_against,
    cohen_kappa,
    discordant_counts,
    mcnemar_exact,
)

LONG_TEXT = "word " * 200


def make_rows(labels):
    return [
        {"text": f"prompt {i}", "label": 0, "label_name": name}
        for i, name in enumerate(labels)
    ]


# ------------------------------------------------------------------- dataset
def test_clip_keeps_short_text_and_cuts_long_text_to_head_and_tail():
    assert clip("  a   b\n c ") == ("a b c", False)
    clipped, cut = clip(LONG_TEXT)
    assert cut is True
    assert "chars cut" in clipped
    assert len(clipped) < len(LONG_TEXT)


def test_rubric_hash_changes_with_the_rubric(tmp_path):
    path = tmp_path / "RUBRIC.md"
    path.write_text("one")
    first = rubric_hash(path)
    path.write_text("two")
    assert first != rubric_hash(path)
    assert len(first) == 8


def test_load_rows_reads_a_split(tmp_path):
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"text": "é", "label_name": "AR"}) + "\n", encoding="utf-8"
    )
    assert load_rows(tmp_path, "test") == [{"text": "é", "label_name": "AR"}]


def test_a_split_is_pinned_on_first_use_and_a_change_is_refused(tmp_path):
    data, manifest = tmp_path / "test.jsonl", tmp_path / "manifest.json"
    data.write_text("row 1\n")
    verify_pinned(data, "test", manifest)
    verify_pinned(data, "test", manifest)  # unchanged file is fine
    data.write_text("row 2\n")
    with pytest.raises(DatasetMismatchError, match="does not match"):
        verify_pinned(data, "test", manifest)
    verify_pinned(data, "validation", manifest)  # another split is pinned separately


# ------------------------------------------------------------------ judgments
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("12 A", Judgment(12, "AR", 0, "H", [])),
        ("13 d", Judgment(13, "DIFFUSION", 0, "H", [])),
        ("14 B M", Judgment(14, "BOTH", 0, "M", [])),
        ("15 A vh dvis", Judgment(15, "AR", 1, "H", ["deliverable_visual"])),
        ("16 A L frag nen", Judgment(16, "AR", 0, "L", ["fragment", "non_english"])),
    ],
)
def test_parse_line(line, expected):
    assert parse_line(line) == expected


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("12", "need"),
        ("x A", "bad id"),
        ("12 Z", "bad label"),
        ("12 A bogus", "unknown flag"),
        ("12 B vh", "vh is only valid"),
    ],
)
def test_parse_line_rejects_bad_input(line, message):
    with pytest.raises(ValueError, match=message):
        parse_line(line)


def test_parse_lines_collects_errors_and_ignores_chatter():
    text = "# header\nHere are the labels:\n0 A\n`1 D`\n1 D\n2 X\n9 A\n"
    parsed, errors, ignored = parse_lines(text, allowed={0, 1, 2})
    assert [j.id for j in parsed] == [0, 1]
    assert ignored == 1
    assert len(errors) == 3  # duplicate id, bad label, id outside the batch


def test_make_record_adds_the_truncated_tag_itself():
    rows = [{"text": LONG_TEXT}, {"text": "short"}]
    long_record = make_record(
        "test", Judgment(0, "AR", 0, "H", ["about_images"]), rows, "j", "r1"
    )
    short_record = make_record("test", Judgment(1, "AR", 0, "H", []), rows, "j", "r1")
    assert long_record["tags"] == ["about_images", "truncated"]
    assert short_record["tags"] == []
    assert long_record["rubric"] == "r1" and long_record["judge"] == "j"


def test_inclusive_label_upgrades_text_only_rows_that_would_benefit_from_visuals():
    assert inclusive_label({"label": "AR", "vh": 1}) == "BOTH"
    assert inclusive_label({"label": "AR", "vh": 0}) == "AR"
    assert inclusive_label({"label": "DIFFUSION", "vh": 0}) == "DIFFUSION"


def test_user_message_lists_one_row_per_id():
    message = build_user_message(make_rows(["AR", "BOTH", "AR"]), [2, 0])
    assert message.splitlines()[-2:] == ["2 prompt 2", "0 prompt 0"]
    assert "Judge these 2 rows" in message


# ----------------------------------------------------------------- checkpoint
def test_checkpoint_round_trip_last_record_wins(tmp_path):
    path = tmp_path / "ck.jsonl"
    assert read_checkpoint(path) == {}
    append_records(path, [{"split": "test", "id": 0, "label": "AR"}])
    append_records(
        path,
        [
            {"split": "test", "id": 0, "label": "BOTH"},
            {"split": "validation", "id": 3, "label": "AR"},
        ],
    )
    records = read_checkpoint(path)
    assert records[("test", 0)]["label"] == "BOTH"
    assert count_judged(records, "test") == 1
    assert list(records_for_split(records, "validation")) == [3]


def test_save_judgments_skips_rows_already_judged_unless_overwriting(tmp_path):
    path, rows = tmp_path / "ck.jsonl", make_rows(["AR", "AR"])
    kwargs = {"split": "test", "judge": "j", "rubric": "r", "path": path}
    first = save_judgments([Judgment(0, "AR", 0, "H", [])], rows, existing={}, **kwargs)
    assert len(first) == 1
    existing = read_checkpoint(path)
    again = save_judgments(
        [Judgment(0, "BOTH", 0, "H", []), Judgment(1, "AR", 0, "H", [])],
        rows,
        existing=existing,
        **kwargs,
    )
    assert [r["id"] for r in again] == [1]
    over = save_judgments(
        [Judgment(0, "BOTH", 0, "H", [])],
        rows,
        existing=read_checkpoint(path),
        overwrite=True,
        **kwargs,
    )
    assert over[0]["label"] == "BOTH"
    assert read_checkpoint(path)[("test", 0)]["label"] == "BOTH"


def test_select_ids_walks_the_unjudged_rows_in_order():
    rows = make_rows(["AR"] * 5)
    primary = {("test", 0): {}, ("test", 2): {}}
    assert select_ids("test", rows, primary, {}, 2, rejudge=False, seed=0) == [1, 3]
    assert select_ids(
        "test", rows, primary, {}, 5, rejudge=False, seed=0, exclude={3}
    ) == [1, 4]


def test_select_ids_for_rejudge_samples_judged_rows_reproducibly():
    rows = make_rows(["AR"] * 6)
    primary = {("test", i): {} for i in range(6)}
    a = select_ids("test", rows, primary, {("test", 0): {}}, 3, rejudge=True, seed=4)
    b = select_ids("test", rows, primary, {("test", 0): {}}, 3, rejudge=True, seed=4)
    assert a == b and 0 not in a and len(a) == 3


# ----------------------------------------------------------------- statistics
def test_kappa_of_perfect_and_chance_agreement():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)


def test_mcnemar_matches_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(15, 9) == pytest.approx(0.307, abs=1e-3)
    assert mcnemar_exact(19, 5) == pytest.approx(0.007, abs=1e-3)
    assert mcnemar_exact(5, 19) == mcnemar_exact(19, 5)


def test_discordant_counts_and_accuracy():
    a, b, ref = ["x", "x", "y", "y"], ["x", "y", "x", "y"], ["x", "x", "x", "y"]
    assert discordant_counts(a, b, ref) == (1, 1)
    assert accuracy_against(a, ref) == 0.75


# ---------------------------------------------------------------- human review
ORIGINAL = ["AR", "AR", "BOTH", "BOTH", "DIFFUSION", "AR"]
RECORDS = {
    i: {"label": label}
    for i, label in enumerate(["AR", "AR", "AR", "BOTH", "DIFFUSION", "BOTH"])
}  # rows 2 and 5 disagree with ORIGINAL


def test_sheet_contains_every_disagreement_and_is_reproducible():
    rows = make_rows(ORIGINAL)
    picked, n_dis = human_review.pick_sheet_rows(rows, RECORDS, n_agree=2, seed=1)
    assert n_dis == 2 and len(picked) == 4 and {2, 5} <= set(picked)
    assert picked == human_review.pick_sheet_rows(rows, RECORDS, n_agree=2, seed=1)[0]


def test_sheet_round_trip_and_split_guard(tmp_path):
    rows, path = make_rows(ORIGINAL), tmp_path / "sheet.tsv"
    human_review.write_sheet(path, rows, [2, 5], "test")
    assert human_review.load_human(str(path), "test") == {}  # nothing filled in yet
    filled = (
        path.read_text()
        .replace("2\tprompt 2\t", "2\tprompt 2\tA")
        .replace("5\tprompt 5\t", "5\tprompt 5\tb")
    )
    path.write_text(filled)
    assert human_review.load_human(str(path), "test") == {2: "AR", 5: "BOTH"}
    with pytest.raises(human_review.SheetSplitError, match="split"):
        human_review.load_human(str(path), "validation")


def test_error_estimate_scales_the_samples_to_the_whole_split():
    human = {
        2: "AR",
        5: "AR",
        0: "AR",
    }  # sides with the judge on row 2 and the original on row 5
    review = human_review.estimate_error_rates(human, ORIGINAL, RECORDS)
    assert review.sides["judge"] == 1 and review.sides["original"] == 1
    assert review.n_disagreement_rows == 2 and review.n_agreement_rows == 1
    # original wrong on 1 of 2 disagreement rows (2 of 6 rows disagree), agreement rows all fine
    assert review.original_error == pytest.approx((2 * (1 / 2)) / 6)
    assert review.judge_error == pytest.approx((2 * (1 / 2)) / 6)


def test_error_estimate_needs_both_strata():
    review = human_review.estimate_error_rates({2: "AR"}, ORIGINAL, RECORDS)
    assert review.original_error is None
    assert "need at least one" in report.human_lines(review)[-1]


# ------------------------------------------------------------------------ report
def test_report_lines_cover_agreement_models_review_and_disagreements():
    rows = make_rows(ORIGINAL)
    records = {
        i: {**r, "conf": "H", "vh": 0, "tags": [], "rubric": "r1"}
        for i, r in RECORDS.items()
    }
    preds = {
        "clean_baseline": ORIGINAL,
        "candidate": ["AR", "AR", "BOTH", "BOTH", "AR", "AR"],
    }
    lines = report.build_report_lines("test", rows, records, preds=preds, show=5)
    text = "\n".join(lines)
    assert "judged 6/6 rows of test" in text
    assert "agreement original vs judge (strict):    4/6" in text
    assert "clean_baseline vs candidate | original" in text
    assert "disagreements (original -> judge)" in text
    assert "   2 BOTH->AR H" in text


def test_load_predictions_drops_a_model_with_the_wrong_length(tmp_path):
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"preds": ["AR", "AR"]}))
    assert report.load_predictions(None, [f"short={path}"], 3) == {}
    assert report.load_predictions(None, [f"ok={path}"], 2) == {"ok": ["AR", "AR"]}


# ------------------------------------------------------------------- API judging
class FakeClient:
    """Answers each request from a script of reply texts."""

    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.requests, self.stop_reason = list(replies), [], stop_reason
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=2),
        )


def run_api(tmp_path, client, rows, **settings):
    path = tmp_path / "ck.jsonl"
    lines = []
    skipped = api_judge.judge_via_api(
        client,
        rows,
        split="test",
        system_text="RUBRIC",
        rubric="r1",
        settings=api_judge.ApiJudgeSettings(model="fake-model", **settings),
        checkpoint=path,
        rejudge_checkpoint=tmp_path / "rj.jsonl",
        target=path,
        out=lines.append,
        err=lines.append,
    )
    return skipped, read_checkpoint(path), lines


def test_api_judging_saves_replies_and_reasks_for_missing_ids(tmp_path):
    rows = make_rows(["AR", "AR", "AR"])
    client = FakeClient(["0 A\n1 D\n", "2 B\n"])
    skipped, records, _lines = run_api(tmp_path, client, rows, batch_size=3)
    assert skipped == set()
    assert {i: r["label"] for (_, i), r in records.items()} == {
        0: "AR",
        1: "DIFFUSION",
        2: "BOTH",
    }
    assert all(
        r["judge"] == "fake-model" and r["rubric"] == "r1" for r in records.values()
    )
    assert len(client.requests) == 2
    assert client.requests[0]["system"][0]["text"] == "RUBRIC"
    assert client.requests[0]["output_config"] == {"effort": "medium"}
    assert "Judge these 1 rows" in client.requests[1]["messages"][0]["content"]


def test_api_judging_gives_up_on_ids_the_model_never_answers(tmp_path):
    rows = make_rows(["AR", "AR"])
    client = FakeClient(["0 A\n", "", ""])
    skipped, records, lines = run_api(tmp_path, client, rows, batch_size=2, retries=1)
    assert skipped == {1}
    assert list(records) == [("test", 0)]
    assert any("giving up" in line for line in lines)


def test_api_judging_honours_the_row_limit(tmp_path):
    rows = make_rows(["AR"] * 4)
    client = FakeClient(["0 A\n1 A\n"])
    _, records, _ = run_api(tmp_path, client, rows, batch_size=2, limit=2)
    assert len(records) == 2 and len(client.requests) == 1


def test_a_reply_cut_off_by_max_tokens_is_an_error():
    client = FakeClient(["0 A"], stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        api_judge.call_model(client, "m", "sys", "user", effort=None, max_tokens=10)


def test_rate_limits_are_retried_then_raised():
    class Flaky:
        def __init__(self, failures):
            self.failures, self.calls = failures, 0
            self.messages = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            self.calls += 1
            if self.calls <= self.failures:
                raise TimeoutError("slow")
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text="0 A")],
                stop_reason="end_turn",
                usage=None,
            )

    waits = []
    text, _ = api_judge.call_model(
        Flaky(2),
        "m",
        "s",
        "u",
        effort=None,
        max_tokens=5,
        retryable=(TimeoutError,),
        sleep=waits.append,
    )
    assert text == "0 A" and waits == [20, 40]
    with pytest.raises(TimeoutError):
        api_judge.call_model(
            Flaky(99),
            "m",
            "s",
            "u",
            effort=None,
            max_tokens=5,
            retryable=(TimeoutError,),
            sleep=lambda s: None,
        )


def test_random_module_is_not_used_implicitly():
    # select_ids must not depend on global random state
    random.seed(1)
    rows = make_rows(["AR"] * 6)
    primary = {("test", i): {} for i in range(6)}
    first = select_ids("test", rows, primary, {}, 3, rejudge=True, seed=2)
    random.seed(999)
    assert first == select_ids("test", rows, primary, {}, 3, rejudge=True, seed=2)
