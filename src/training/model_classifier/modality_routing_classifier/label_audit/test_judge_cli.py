"""End-to-end tests of the judge_labels command line on a tiny synthetic dataset."""

import io
import json
import sys

import judge_labels
import pytest
from audit_lib.dataset import text_sha256

LABELS = ["AR", "AR", "DIFFUSION", "BOTH", "BOTH", "AR"]


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"text": f"prompt number {i}", "label": 0, "label_name": name}
        for i, name in enumerate(LABELS)
    ]
    (data / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def run(workspace, capsys, *argv, stdin=None, monkeypatch=None):
    common = [
        "--data-dir",
        str(workspace / "data"),
        "--checkpoint",
        str(workspace / "ck.jsonl"),
        "--rejudge-checkpoint",
        str(workspace / "rj.jsonl"),
    ]
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = judge_labels.main([argv[0], *common, *argv[1:]])
    return code, capsys.readouterr().out


def test_next_save_status_report_and_sheet_flow(workspace, capsys, monkeypatch):
    code, out = run(workspace, capsys, "next", "--n", "4")
    assert code == 0
    assert out.splitlines()[0].startswith("# split=test remaining=6 showing=4")
    assert (
        "0 prompt number 0" in out and "original" not in out
    )  # blinded: no labels shown

    code, out = run(
        workspace,
        capsys,
        "save",
        stdin="0 A\n1 A\n2 D\n3 A vh\n99 A\n",
        monkeypatch=monkeypatch,
    )
    assert code == 1  # one rejected line (id out of range)
    assert "saved 4" in out and "test: 4/6 judged" in out

    code, out = run(
        workspace, capsys, "save", stdin="3 D\n4 B\n5 A\n", monkeypatch=monkeypatch
    )
    assert code == 0
    assert (
        "saved 2" in out and "already judged (skipped) 1" in out
    )  # row 3 kept its first judgment

    code, out = run(workspace, capsys, "status")
    assert "test: 6/6 judged" in out

    code, out = run(workspace, capsys, "report", "--show", "5")
    assert code == 0
    assert "agreement original vs judge (strict):    5/6" in out
    assert "3 BOTH->AR" in out

    sheet = workspace / "sheet.tsv"
    code, out = run(workspace, capsys, "sheet", "--out", str(sheet), "--n-agree", "2")
    assert code == 0 and "1 disagreements + 2 sampled agreements" in out
    assert sheet.read_text().splitlines()[0] == "# split=test"


def test_report_before_any_judgment_says_so(workspace, capsys):
    code, out = run(workspace, capsys, "report")
    assert code == 1 and "no judgments yet" in out


def test_a_human_sheet_for_another_split_is_refused(workspace, capsys, monkeypatch):
    run(
        workspace,
        capsys,
        "save",
        stdin="0 A\n1 A\n2 D\n3 A\n4 B\n5 A\n",
        monkeypatch=monkeypatch,
    )
    sheet = workspace / "sheet.tsv"
    run(workspace, capsys, "sheet", "--out", str(sheet))
    sheet.write_text(sheet.read_text().replace("# split=test", "# split=validation"))
    code = judge_labels.main(
        [
            "report",
            "--data-dir",
            str(workspace / "data"),
            "--checkpoint",
            str(workspace / "ck.jsonl"),
            "--rejudge-checkpoint",
            str(workspace / "rj.jsonl"),
            "--human",
            str(sheet),
        ]
    )
    assert code == 1
    assert "made for split" in capsys.readouterr().err


def test_api_dry_run_shows_the_first_batch_without_calling_the_api(workspace, capsys):
    code, out = run(workspace, capsys, "api", "--dry-run", "--batch-size", "2")
    assert code == 0
    assert "batch=2" in out and "0 prompt number 0" in out


def rewrite_split(workspace, labels):
    rows = [
        {"text": f"changed prompt {i}", "label": 0, "label_name": name}
        for i, name in enumerate(labels)
    ]
    (workspace / "data" / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def judged_workspace(workspace, capsys, monkeypatch):
    run(
        workspace,
        capsys,
        "save",
        stdin="0 A\n1 A\n2 D\n3 A\n4 B\n5 A\n",
        monkeypatch=monkeypatch,
    )


def test_a_custom_data_dir_is_pinned_beside_its_checkpoint(
    workspace, capsys, monkeypatch
):
    judged_workspace(workspace, capsys, monkeypatch)
    manifest = workspace / "ck.jsonl.dataset_sha256.json"
    assert list(json.loads(manifest.read_text())) == ["test"]


def test_replacing_a_custom_split_after_judging_is_refused(
    workspace, capsys, monkeypatch
):
    judged_workspace(workspace, capsys, monkeypatch)
    rewrite_split(workspace, LABELS)
    for command in ("report", "next", "sheet"):
        code, _out = run(workspace, capsys, command)
        assert code == 1, command


def test_reordering_a_custom_split_after_judging_is_refused(
    workspace, capsys, monkeypatch
):
    judged_workspace(workspace, capsys, monkeypatch)
    path = workspace / "data" / "test.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    code = judge_labels.main(
        [
            "report",
            "--data-dir",
            str(workspace / "data"),
            "--checkpoint",
            str(workspace / "ck.jsonl"),
            "--rejudge-checkpoint",
            str(workspace / "rj.jsonl"),
        ]
    )
    assert code == 1
    assert (
        "does not match the file the test judgments were made on"
        in capsys.readouterr().err
    )


def test_judgments_with_no_recorded_dataset_hash_are_refused(
    workspace, capsys, monkeypatch
):
    judged_workspace(workspace, capsys, monkeypatch)
    (
        workspace / "ck.jsonl.dataset_sha256.json"
    ).unlink()  # e.g. a checkpoint from before pinning
    code, _ = run(workspace, capsys, "report")
    assert code == 1


def test_export_judged_keeps_the_original_label_as_a_second_column(
    workspace, capsys, monkeypatch
):
    judged_workspace(workspace, capsys, monkeypatch)
    out_path = workspace / "judged_test.jsonl"
    code, out = run(workspace, capsys, "export-judged", "--out", str(out_path))
    assert code == 0
    assert "wrote 6 rows" in out
    rows = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["id"] for r in rows] == [0, 1, 2, 3, 4, 5]
    # row 2 was originally DIFFUSION; "2 D" was judged the same, so nothing moved
    assert rows[2]["label_name"] == "DIFFUSION"
    assert rows[2]["judged_label_name"] == "DIFFUSION"
    # every original field survives untouched beside the judged_* ones
    for row in rows:
        assert row["label_name"] == LABELS[row["id"]]


def test_export_judged_before_any_judging_is_refused(workspace, capsys):
    code, out = run(workspace, capsys, "export-judged")
    assert code == 1 and "no judgments yet" in out


def test_report_human_adds_a_corrected_mcnemar_line(workspace, capsys, monkeypatch):
    judged_workspace(workspace, capsys, monkeypatch)
    # Human corrects row 3 back to BOTH (the judge said A); everything else stands.
    sheet = workspace / "sheet.tsv"
    run(workspace, capsys, "sheet", "--out", str(sheet), "--n-agree", "5")
    filled = sheet.read_text(encoding="utf-8")
    for row_id, label in [(0, "A"), (1, "A"), (2, "D"), (3, "B"), (4, "B"), (5, "A")]:
        filled = filled.replace(
            f"{row_id}\tprompt number {row_id}\t",
            f"{row_id}\tprompt number {row_id}\t{label}",
        )
    sheet.write_text(filled, encoding="utf-8")

    hashes = [text_sha256(f"prompt number {i}") for i in range(len(LABELS))]
    clean = [
        "AR",
        "AR",
        "DIFFUSION",
        "AR",
        "BOTH",
        "AR",
    ]  # wrong on row 3 under both policies
    candidate = [
        "AR",
        "AR",
        "DIFFUSION",
        "BOTH",
        "BOTH",
        "AR",
    ]  # right on row 3 once corrected
    preds_path = workspace / "preds.json"
    preds_path.write_text(
        json.dumps({"preds": candidate, "input_hashes": hashes}), encoding="utf-8"
    )
    eval_report = workspace / "eval_report.json"
    eval_report.write_text(
        json.dumps(
            {
                "per_example_records": [
                    {
                        "row_index": i,
                        "input_hash_sha256": hashes[i],
                        "clean_baseline_pred": clean[i],
                    }
                    for i in range(len(LABELS))
                ]
            }
        ),
        encoding="utf-8",
    )
    code, out = run(
        workspace,
        capsys,
        "report",
        "--eval-report",
        str(eval_report),
        "--preds",
        f"candidate={preds_path}",
        "--human",
        str(sheet),
    )
    assert code == 0
    assert "clean_baseline vs candidate | corrected" in out
    # under "strict" (the judge's own label), row 3 favours clean_baseline (right where candidate is wrong);
    # once the human correction is applied to row 3, candidate instead picks up that row.
    strict_line = next(line for line in out.splitlines() if "| strict" in line)
    corrected_line = next(line for line in out.splitlines() if "| corrected" in line)
    assert "only clean_baseline right 1" in strict_line
    assert "only candidate right 1" in corrected_line
