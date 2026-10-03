import importlib
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

signal_labeled_set = importlib.import_module("signal_labeled_set")

PREVIEW_ACTIONS = {
    "Fix the crash.": ["fix"],
    "Why is it slow?": ["explain"],
    "Write tests and fix it.": ["test"],
    "yes": [],
}


class PreviewHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        PreviewHandler.requests.append({"path": self.path, "body": body})
        text = body["messages"][-1]["content"]
        payload = json.dumps(
            {"decision_result": {"matched_signals": {"action": PREVIEW_ACTIONS[text]}}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


class ScoreTest(unittest.TestCase):
    def test_score_reports_precision_recall_and_confusion(self) -> None:
        report = signal_labeled_set.score(
            [
                ("fix", "fix"),
                ("fix", "explain"),
                ("explain", "explain"),
                ("other", signal_labeled_set.NO_MATCH),
            ],
            ["fix", "explain", "other"],
        )

        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["per_label"]["fix"]["precision"], 1.0)
        self.assertEqual(report["per_label"]["fix"]["recall"], 0.5)
        self.assertEqual(report["per_label"]["explain"]["precision"], 0.5)
        self.assertEqual(report["per_label"]["explain"]["recall"], 1.0)
        self.assertEqual(report["per_label"]["other"]["recall"], 0.0)
        self.assertEqual(
            report["columns"], ["fix", "explain", "other", signal_labeled_set.NO_MATCH]
        )
        self.assertEqual(report["confusion"]["fix"]["explain"], 1)
        self.assertEqual(report["confusion"]["other"][signal_labeled_set.NO_MATCH], 1)

    def test_predicted_label_rejects_several_labels(self) -> None:
        response = {"decision_result": {"matched_signals": {"action": ["fix", "test"]}}}
        with self.assertRaises(ValueError):
            signal_labeled_set.predicted_label(response, "action")


class PreviewScoringTest(unittest.TestCase):
    def test_main_scores_each_prompt_through_the_preview_route(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), PreviewHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        PreviewHandler.requests = []
        records = [
            {"text": "Fix the crash.", "true_label": "fix"},
            {"text": "Why is it slow?", "true_label": "explain"},
            {"text": "Write tests and fix it.", "true_label": "fix"},
            {"text": "yes", "true_label": "other"},
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            dataset = Path(tempdir) / "labeled.json"
            dataset.write_text(json.dumps(records), encoding="utf-8")
            output = Path(tempdir) / "report.json"
            exit_code = signal_labeled_set.main(
                [
                    "--router-url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--dataset",
                    str(dataset),
                    "--json-output",
                    str(output),
                ]
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {request["path"] for request in PreviewHandler.requests},
            {signal_labeled_set.PREVIEW_PATH},
        )
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["confusion"]["fix"]["test"], 1)
        self.assertEqual(report["confusion"]["other"][signal_labeled_set.NO_MATCH], 1)
        self.assertEqual(
            [result["predicted"] for result in report["results"]],
            ["fix", "explain", "test", signal_labeled_set.NO_MATCH],
        )


if __name__ == "__main__":
    unittest.main()
