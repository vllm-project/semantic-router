"""Real HTTP judge acquisition preserves failures, provenance and prompt bytes."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from systemone_auto.artifacts import (
    canonical,
    digest,
    file_digest,
    read_json,
    write_json,
)
from systemone_auto.judge import collect_judge, score_judge, selection, validate_inputs
from test_collection import pilot
from test_contracts import response

FIXTURE = Path(__file__).parent / "fixtures/judge-request.json"


def test_go_prompt_golden_preserves_utf8_content():
    golden = read_json(FIXTURE)
    assert [
        hashlib.sha256(m["content"].encode()).hexdigest()
        for m in golden["request"]["messages"]
    ] == golden["message_content_sha256"]
    data = json.loads(golden["request"]["messages"][1]["content"])
    assert data["request"] == golden["source"]["request"]
    assert data["candidates"] == {
        c["stage"]: c["body"] for c in golden["source"]["candidates"]
    }
    assert "labels" not in data


@pytest.mark.parametrize(
    "content,finish,outcome",
    [
        ("{}", "stop", "invalid_selection"),
        ('{"selected":"abstain"}', "stop", "abstained"),
        ('{"selected":"upgrade","confidence":1}', "stop", "invalid_selection"),
        ('{"selected":"upgrade"}', "length", "incomplete"),
    ],
)
def test_judge_rejects_invented_or_incomplete_selection(content, finish, outcome):
    raw = {"choices": [{"finish_reason": finish, "message": {"content": content}}]}
    assert selection(raw, ["fast", "upgrade"]) == (None, outcome)


def test_judge_collection_resume_and_failure_denominator(tmp_path, monkeypatch):
    data = pilot()
    golden = read_json(FIXTURE)
    rows = []
    for source in data["records"]:
        candidates = {"fast": response(source), "upgrade": response(source)}
        payload = copy.deepcopy(golden["request"])
        payload["messages"][1]["content"] = canonical(
            {"request": source["request"], "candidates": candidates}
        )
        rows.append(
            {
                "record_id": source["id"],
                "group_id": source["group_id"],
                "split": source["split"],
                "request": payload,
                "judge_request_sha256": digest(payload),
                "original_request_sha256": digest(source["request"]),
                "native_candidate_sha256": {
                    k: digest(v) for k, v in candidates.items()
                },
            }
        )
    inputs = tmp_path / "requests.jsonl"
    inputs.write_text("".join(canonical(r) + "\n" for r in rows))
    manifest = {
        "schema_version": "systemone-judge-inputs/v1",
        "request_builder": "systemone.JudgeRequest",
        "requests": len(rows),
        "requests_sha256": file_digest(inputs),
        "dataset_sha256": digest(data),
        "served_model_id": golden["served_model_id"],
        "system_prompt_sha256": digest(golden["request"]["messages"][0]["content"]),
        "candidate_stages": ["fast", "upgrade"],
        "model_id": "Qwen/Qwen3.8-Flash-Next",
        "revision": "a" * 40,
    }
    validate_inputs(rows, manifest)
    changed = copy.deepcopy(rows)
    changed[0]["request"]["messages"][1]["content"] += " "
    with pytest.raises(ValueError, match="digest"):
        validate_inputs(changed, manifest)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.headers.get("Authorization"))
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert (
                payload["max_tokens"] == 128 and "chat_template_kwargs" not in payload
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            selected = "abstain" if len(calls) == len(rows) else "upgrade"
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps({"selected": selected})
                                },
                            }
                        ]
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        target = {
            "name": "flashnext",
            "protocol": "chat",
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "endpoint": f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            "api_key_env": "JUDGE_TEST_KEY",
        }
        monkeypatch.setenv("JUDGE_TEST_KEY", "judge-secret")
        deployment = {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "default_chat_template_kwargs": {"enable_thinking": False},
        }
        paths = [
            tmp_path / name
            for name in (
                "manifest.json",
                "target.json",
                "deployment.json",
                "dataset.json",
            )
        ]
        for path, value in zip(
            paths, (manifest, target, deployment, data), strict=True
        ):
            write_json(path, value)
        output = tmp_path / "output"
        first = collect_judge(inputs, *paths[:3], output, limit=3)
        assert first["observation_count"] == 3 and not first["complete"]
        with pytest.raises(ValueError, match="complete"):
            score_judge(paths[3], inputs, output)
        final = collect_judge(inputs, *paths[:3], output)
        assert final["complete"] and len(calls) == 8
        collect_judge(inputs, *paths[:3], output)
        assert len(calls) == 8 and all(c == "Bearer judge-secret" for c in calls)
        contents = "".join(p.read_text() for p in output.iterdir())
        assert "judge-secret" not in contents and "127.0.0.1" not in contents
        score = score_judge(paths[3], inputs, output)
        assert score["outcomes"] == {"selected": 7, "abstained": 1}
        assert score["splits"]["held_out"]["bundle_accuracy"] == 0.5
        deployment["default_chat_template_kwargs"]["enable_thinking"] = True
        write_json(paths[2], deployment)
        with pytest.raises(ValueError, match="nonthinking"):
            collect_judge(inputs, *paths[:3], output)
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()
