"""The pinned export source tolerates transient Hub failures without drift."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contract import digest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests import Response
from requests.exceptions import ReadTimeout
from source import HUB_METADATA_TIMEOUT_SECONDS, HUB_SNAPSHOT_ATTEMPTS, provision


def wrapped_hub_error(cause: Exception) -> LocalEntryNotFoundError:
    error = LocalEntryNotFoundError("snapshot is not cached")
    error.__cause__ = cause
    return error


def http_error(status: int) -> HfHubHTTPError:
    response = Response()
    response.status_code = status
    return HfHubHTTPError(f"Hub returned {status}", response=response)


class SourceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.snapshot = Path(self.temporary.name)
        data = self.snapshot / "model.safetensors"
        data.write_bytes(b"pinned model bytes")
        self.pinned = {
            "repo_id": "example/pinned-model",
            "revision": "a" * 40,
            "files": {
                data.name: {
                    "algorithm": "sha256",
                    "digest": digest(data),
                    "size": data.stat().st_size,
                }
            },
        }
        self.sources_patch = patch("source.sources", return_value={"nano": self.pinned})
        self.sources_patch.start()
        self.addCleanup(self.sources_patch.stop)

    def test_export_process_sets_transfer_timeout_before_hub_import(self):
        environment = os.environ.copy()
        environment.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import source; from huggingface_hub import constants; "
                "assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 60",
            ],
            cwd=Path(__file__).parent,
            env=environment,
            check=True,
            capture_output=True,
        )

    def test_retries_wrapped_504_and_preserves_pinned_inputs(self):
        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=[
                    wrapped_hub_error(http_error(504)),
                    str(self.snapshot),
                ],
            ) as download,
            patch("source.random.uniform", return_value=0),
            patch("source.time.sleep") as sleep,
        ):
            self.assertEqual(provision("nano", None, True), self.snapshot.resolve())

        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0], download.call_args_list[1])
        self.assertEqual(download.call_args.kwargs["revision"], self.pinned["revision"])
        self.assertEqual(
            download.call_args.kwargs["allow_patterns"], ["model.safetensors"]
        )
        self.assertEqual(
            download.call_args.kwargs["etag_timeout"], HUB_METADATA_TIMEOUT_SECONDS
        )
        sleep.assert_called_once_with(1)

    def test_retries_wrapped_read_timeout_and_stops_after_bounded_attempts(self):
        failure = wrapped_hub_error(ReadTimeout("metadata request timed out"))
        with (
            patch("huggingface_hub.snapshot_download", side_effect=failure) as download,
            patch("source.random.uniform", return_value=0),
            patch("source.time.sleep") as sleep,
            self.assertRaises(LocalEntryNotFoundError),
        ):
            provision("nano", None, True)

        self.assertEqual(download.call_count, HUB_SNAPSHOT_ATTEMPTS)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2, 4])

    def test_does_not_retry_missing_revision_or_identity_mismatch(self):
        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=wrapped_hub_error(http_error(404)),
            ) as download,
            patch("source.time.sleep") as sleep,
            self.assertRaises(LocalEntryNotFoundError),
        ):
            provision("nano", None, True)
        download.assert_called_once()
        sleep.assert_not_called()

        with patch(
            "huggingface_hub.snapshot_download", return_value=str(self.snapshot)
        ):
            (self.snapshot / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                provision("nano", None, True)


if __name__ == "__main__":
    unittest.main()
