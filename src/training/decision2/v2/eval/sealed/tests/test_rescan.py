from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

SEALED = Path(__file__).resolve().parents[1]
SCRIPT = SEALED / "c1-rescan.sh"
SPEC = SEALED / "c1-rescan-coverage.json"
EVENT = SEALED / "event3.sh"


def pinned(text: str, name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", text, re.MULTILINE)
    assert match, name
    return match.group(1)


class RescanSpecTest(unittest.TestCase):
    spec = json.loads(SPEC.read_text())
    event = EVENT.read_text()

    def test_pins_match_the_event_script(self) -> None:
        item_set = pinned(self.event, "ITEM_SET")
        entry = self.spec["item_sets"][item_set]
        retired = pinned(self.event, "RETIRED").replace(
            "$C1", "/data/dev2/private/sealed/c1"
        )
        self.assertEqual(entry["retired"], retired)
        self.assertEqual(entry["retired_sha256"], pinned(self.event, "RETIRED_SHA"))
        self.assertEqual(
            self.spec["protected_sha256"], pinned(self.event, "PROTECTED_SHA")
        )
        folder = "c1-rescan-" + item_set.replace(".", "_")
        self.assertEqual(
            pinned(self.event, "SCAN_VERDICT"),
            f"/data/dev2/runs/eval/m4/{folder}/SCAN-VERDICT.json",
        )

    def test_event2_baseline_and_scan_settings(self) -> None:
        self.assertEqual(
            self.spec["baseline"]["sha256"],
            "077f1b53148c1fa4bc1a84d8caebba156c5da800a8d86e8989148deeef6984dd",
        )
        self.assertEqual(
            self.spec["image"],
            "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54",
        )
        self.assertEqual(
            self.spec["scan"], ["--workers", "48", "--exact-min-tokens", "8"]
        )
        self.assertEqual(self.spec["hf_delta"]["base"][:8], "30e0a1f7")

    def test_coverage_roots_and_excludes(self) -> None:
        for root in ("/data/dev2/runs", "/data/dev2/private", "/data/dev2/hf-cache"):
            self.assertIn(root, self.spec["roots"])
        for path in (
            "/data/dev2/private/sealed",
            "/data/dev2/private/panels",
            "/data/dev2/private/htdev",
            "/data/dev2/private/eval",
            "/data/dev2/runs/eval",
            "/data/dev2/src",
        ):
            self.assertIn(path, self.spec["excludes"])
        self.assertNotIn("/data/dev2/hf-cache/blobs", self.spec["excludes"])


class RescanScriptTest(unittest.TestCase):
    text = SCRIPT.read_text()

    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        if shutil.which("shellcheck"):
            subprocess.run(["shellcheck", "-S", "warning", str(SCRIPT)], check=True)

    def test_containers_are_offline_cpu_only_and_blind_to_the_sealed_dir(self) -> None:
        runs = [m.start() for m in re.finditer(r"docker run ", self.text)]
        self.assertEqual(len(runs), 2)
        for start in runs:
            command = self.text[start : self.text.index('"$IMG"', start)]
            self.assertIn("--network none", command)
            self.assertIn('"${MOUNTS[@]}"', command)
        self.assertNotIn("/dev/kfd", self.text)
        self.assertNotIn("--device", self.text)
        tmpfs = self.text.index(
            'MOUNTS+=(--mount "type=tmpfs,destination=/data/dev2/private/sealed")'
        )
        self.assertLess(tmpfs, runs[0])

    def test_checks_precede_the_scan(self) -> None:
        scan = self.text.index("-m v2.eval.sealed.overlap scan")
        for check in (
            '[ "$(sha "$PROT")" = "$PROT_SHA" ]',
            "docker image inspect",
            '>>"$ACCESS"',
            'RETIRED_SHA" =~ ^[0-9a-f]{64}$',
        ):
            self.assertLess(self.text.index(check), scan)

    def test_judge_checks_pins_before_extracting(self) -> None:
        judge = self.text.index('[ "$CMD" = judge ]')
        extract = self.text.index("scanverdict extract")
        for check in (
            '[ "$(sha "$BASE")" = "$BASE_SHA" ]',
            '[ "$(sha "$RETIRED")" = "$RETIRED_SHA" ]',
        ):
            self.assertLess(judge, self.text.index(check))
            self.assertLess(self.text.index(check), extract)
        self.assertIn(
            '--retired-sha "$RETIRED_SHA" --protected-sha "$PROT_SHA"', self.text
        )
        self.assertIn('cp "$J/SCAN-VERDICT.json" "$W/SCAN-VERDICT.json"', self.text)

    def test_every_spec_key_the_script_reads_resolves(self) -> None:
        start = self.text.index("spec() { python3 -c '") + len("spec() { python3 -c '")
        code = self.text[start : self.text.index('\' "$SPEC" "$1"; }', start)]
        keys = re.findall(r'spec "?([a-z_0-9/$]+(?:\$SET)?[a-z_0-9/]*)"?\)', self.text)
        self.assertGreaterEqual(len(keys), 12)
        for key in keys:
            key = key.replace("$SET", "v1.2")
            result = subprocess.run(
                ["python3", "-c", code, str(SPEC), key],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, (key, result.stderr))
            self.assertTrue(result.stdout.strip(), key)

    def test_hf_delta_lands_under_a_covered_root(self) -> None:
        self.assertIn('H="/data/dev2/private/c1-rescan-hf-delta/', self.text)
        self.assertIn("--token-file /root/.cache/huggingface/token", self.text)


if __name__ == "__main__":
    unittest.main()
