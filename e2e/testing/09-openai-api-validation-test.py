#!/usr/bin/env python3
"""
E2E OpenAI API validation test suite.

Validates that the OpenAI API contract (Files, Vector Stores, Vector Store Files,
Vector Store Search) used by semantic-router remains compatible with upstream.
Adapted from test patterns in https://github.com/openai/openai-python/tree/main/tests.

Run against the real OpenAI API when OPENAI_API_KEY is set. Skips all tests when
the key is not set. Use OPENAI_BASE_URL to override the API base (default:
https://api.openai.com/v1).
"""

import os
import sys
import json
import time
import tempfile
import requests
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OpenAIAPIValidationTest:
    """Validate OpenAI API contract (Files, Vector Stores, Search) against upstream."""

    def log(self, msg: str) -> None:
        """Print message."""
        print(msg)

    def __init__(self, base_url: Optional[str] = None, verbose: bool = True):
        # base_url ignored when validating against OpenAI directly; kept for CLI compat
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self._created_file_id: Optional[str] = None
        self._created_vector_store_id: Optional[str] = None

    # Placeholder/example keys; skip validation when these are set (avoid 401s).
    _PLACEHOLDER_KEYS = (
        "sk-your-key",
        "sk-your-real-key",
        "sk-...",
        "your-api-key",
    )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _skip_no_key(self) -> bool:
        if not self.api_key:
            self.log(
                "SKIP: OPENAI_API_KEY not set. Set it to run OpenAI API validation tests."
            )
            return True
        key_lower = self.api_key.strip().lower()
        for placeholder in self._PLACEHOLDER_KEYS:
            if key_lower == placeholder:
                self.log(
                    "SKIP: OPENAI_API_KEY looks like a placeholder. Set a real key to run OpenAI API validation tests."
                )
                return True
        if len(self.api_key) < 20:
            self.log(
                "SKIP: OPENAI_API_KEY too short (likely placeholder). Set a real key to run OpenAI API validation tests."
            )
            return True
        return False

    # --- Files API (adapted from openai-python tests/api_resources) ---

    def test_files_list(self) -> bool:
        """List files - validate response schema (object, data, has_more)."""
        self.log("[OpenAI API Validation] Files: list...")
        if self._skip_no_key():
            return True
        try:
            r = requests.get(
                f"{self.base_url}/files",
                headers=self._headers(),
                timeout=30,
            )
            if r.status_code != 200:
                self.log(f"ERROR: list files status {r.status_code}: {r.text}")
                return False
            data = r.json()
            if "object" not in data:
                self.log("ERROR: response missing 'object'")
                return False
            if "data" not in data:
                self.log("ERROR: response missing 'data'")
                return False
            if "has_more" not in data:
                self.log("ERROR: response missing 'has_more'")
                return False
            for f in data.get("data", [])[:3]:
                if "id" not in f or "object" not in f:
                    self.log("ERROR: file item missing id or object")
                    return False
            self.log("[OpenAI API Validation] Files list: schema OK")
            return True
        except Exception as e:
            self.log(f"ERROR: Files list failed: {e}")
            return False

    def test_files_upload_get_delete(self) -> bool:
        """Upload file, get file, delete file - full lifecycle."""
        self.log("[OpenAI API Validation] Files: upload -> get -> delete...")
        if self._skip_no_key():
            return True
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
            ) as tmp:
                tmp.write("OpenAI API validation test content.\n")
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    r = requests.post(
                        f"{self.base_url}/files",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files={"file": ("validation_test.txt", f, "text/plain")},
                        data={"purpose": "assistants"},
                        timeout=60,
                    )
                if r.status_code != 200:
                    self.log(f"ERROR: upload status {r.status_code}: {r.text}")
                    return False
                file_obj = r.json()
                if "id" not in file_obj or "object" not in file_obj:
                    self.log("ERROR: upload response missing id or object")
                    return False
                file_id = file_obj["id"]
                self._created_file_id = file_id

                # Get file
                r2 = requests.get(
                    f"{self.base_url}/files/{file_id}",
                    headers=self._headers(),
                    timeout=30,
                )
                if r2.status_code != 200:
                    self.log(f"ERROR: get file status {r2.status_code}: {r2.text}")
                    return False
                get_obj = r2.json()
                if get_obj.get("id") != file_id:
                    self.log("ERROR: get file id mismatch")
                    return False

                # Delete file
                r3 = requests.delete(
                    f"{self.base_url}/files/{file_id}",
                    headers=self._headers(),
                    timeout=30,
                )
                if r3.status_code != 200:
                    self.log(f"ERROR: delete file status {r3.status_code}: {r3.text}")
                    return False
                self._created_file_id = None
                self.log("[OpenAI API Validation] Files upload/get/delete: OK")
                return True
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            self.log(f"ERROR: Files lifecycle failed: {e}")
            return False

    # --- Vector Stores API ---

    def test_vector_stores_list(self) -> bool:
        """List vector stores - validate schema."""
        self.log("[OpenAI API Validation] Vector Stores: list...")
        if self._skip_no_key():
            return True
        try:
            r = requests.get(
                f"{self.base_url}/vector_stores",
                headers=self._headers(),
                params={"limit": 5},
                timeout=30,
            )
            if r.status_code != 200:
                self.log(f"ERROR: list vector stores status {r.status_code}: {r.text}")
                return False
            data = r.json()
            if "object" not in data or "data" not in data:
                self.log("ERROR: response missing object or data")
                return False
            for vs in data.get("data", [])[:2]:
                if "id" not in vs or "object" not in vs:
                    self.log("ERROR: vector_store item missing id or object")
                    return False
            self.log("[OpenAI API Validation] Vector Stores list: schema OK")
            return True
        except Exception as e:
            self.log(f"ERROR: Vector Stores list failed: {e}")
            return False

    def test_vector_stores_create_get_update_delete(self) -> bool:
        """Create, get, update, delete vector store - full lifecycle."""
        self.log(
            "[OpenAI API Validation] Vector Stores: create -> get -> update -> delete..."
        )
        if self._skip_no_key():
            return True
        try:
            r = requests.post(
                f"{self.base_url}/vector_stores",
                headers=self._headers(),
                json={"name": "e2e-validation-test-vs"},
                timeout=30,
            )
            if r.status_code != 200:
                self.log(f"ERROR: create vector store status {r.status_code}: {r.text}")
                return False
            vs = r.json()
            if "id" not in vs or "object" not in vs:
                self.log("ERROR: create response missing id or object")
                return False
            vs_id = vs["id"]
            self._created_vector_store_id = vs_id

            # Get
            r2 = requests.get(
                f"{self.base_url}/vector_stores/{vs_id}",
                headers=self._headers(),
                timeout=30,
            )
            if r2.status_code != 200:
                self.log(f"ERROR: get vector store status {r2.status_code}: {r2.text}")
                return False
            if r2.json().get("id") != vs_id:
                self.log("ERROR: get vector store id mismatch")
                return False

            # Update (name)
            r3 = requests.post(
                f"{self.base_url}/vector_stores/{vs_id}",
                headers=self._headers(),
                json={"name": "e2e-validation-test-vs-updated"},
                timeout=30,
            )
            if r3.status_code != 200:
                self.log(
                    f"ERROR: update vector store status {r3.status_code}: {r3.text}"
                )
                return False

            # Delete
            r4 = requests.delete(
                f"{self.base_url}/vector_stores/{vs_id}",
                headers=self._headers(),
                timeout=30,
            )
            if r4.status_code != 200:
                self.log(
                    f"ERROR: delete vector store status {r4.status_code}: {r4.text}"
                )
                return False
            self._created_vector_store_id = None
            self.log(
                "[OpenAI API Validation] Vector Stores create/get/update/delete: OK"
            )
            return True
        except Exception as e:
            self.log(f"ERROR: Vector Stores lifecycle failed: {e}")
            return False

    # --- Vector Store Files ---

    def test_vector_store_files_lifecycle(self) -> bool:
        """Create vector store, add file (if we have one), list files, delete."""
        self.log(
            "[OpenAI API Validation] Vector Store Files: create vs -> list files..."
        )
        if self._skip_no_key():
            return True
        vs_id = None
        try:
            r = requests.post(
                f"{self.base_url}/vector_stores",
                headers=self._headers(),
                json={"name": "e2e-validation-vs-files"},
                timeout=30,
            )
            if r.status_code != 200:
                self.log(f"ERROR: create vs for files status {r.status_code}: {r.text}")
                return False
            vs_id = r.json()["id"]

            # List vector store files (may be empty)
            r2 = requests.get(
                f"{self.base_url}/vector_stores/{vs_id}/files",
                headers=self._headers(),
                params={"limit": 10},
                timeout=30,
            )
            if r2.status_code != 200:
                self.log(
                    f"ERROR: list vector store files status {r2.status_code}: {r2.text}"
                )
                return False
            data = r2.json()
            if "object" not in data or "data" not in data:
                self.log("ERROR: list vector store files missing object or data")
                return False

            # Delete vector store
            requests.delete(
                f"{self.base_url}/vector_stores/{vs_id}",
                headers=self._headers(),
                timeout=30,
            )
            self.log("[OpenAI API Validation] Vector Store Files list: OK")
            return True
        except Exception as e:
            self.log(f"ERROR: Vector Store Files lifecycle failed: {e}")
            return False
        finally:
            if vs_id:
                try:
                    requests.delete(
                        f"{self.base_url}/vector_stores/{vs_id}",
                        headers=self._headers(),
                        timeout=30,
                    )
                except Exception:
                    pass

    # --- Vector Store Search ---

    def test_vector_store_search_schema(self) -> bool:
        """Vector store search - validate request/response schema (query, object, data)."""
        self.log("[OpenAI API Validation] Vector Store Search: schema check...")
        if self._skip_no_key():
            return True
        vs_id = None
        try:
            r = requests.post(
                f"{self.base_url}/vector_stores",
                headers=self._headers(),
                json={"name": "e2e-validation-search"},
                timeout=30,
            )
            if r.status_code != 200:
                self.log(
                    f"ERROR: create vs for search status {r.status_code}: {r.text}"
                )
                return False
            vs_id = r.json()["id"]
            # Wait for vector store to be ready (empty vs is quickly ready)
            time.sleep(2)

            search_r = requests.post(
                f"{self.base_url}/vector_stores/{vs_id}/search",
                headers=self._headers(),
                json={"query": "test query"},
                timeout=30,
            )
            if search_r.status_code != 200:
                self.log(
                    f"ERROR: search status {search_r.status_code}: {search_r.text}"
                )
                return False
            data = search_r.json()
            if "object" not in data:
                self.log("ERROR: search response missing object")
                return False
            if "data" not in data:
                self.log("ERROR: search response missing data")
                return False
            # Each result should have content, filename (optional), score (optional)
            for item in data.get("data", [])[:3]:
                if "content" not in item:
                    self.log("ERROR: search result missing content")
                    return False
            self.log("[OpenAI API Validation] Vector Store Search: schema OK")
            return True
        except Exception as e:
            self.log(f"ERROR: Vector Store Search failed: {e}")
            return False
        finally:
            if vs_id:
                try:
                    requests.delete(
                        f"{self.base_url}/vector_stores/{vs_id}",
                        headers=self._headers(),
                        timeout=30,
                    )
                except Exception:
                    pass

    def run_all_tests(self) -> bool:
        """Run all OpenAI API validation tests."""
        self.log("=" * 60)
        self.log("OpenAI API Validation E2E Test Suite")
        self.log("Adapted from openai-python/tests")
        self.log("=" * 60)
        if self._skip_no_key():
            self.log("All tests skipped.")
            return True

        results: List[tuple] = []
        results.append(("Files list", self.test_files_list()))
        results.append(("Files upload/get/delete", self.test_files_upload_get_delete()))
        results.append(("Vector Stores list", self.test_vector_stores_list()))
        results.append(
            (
                "Vector Stores create/get/update/delete",
                self.test_vector_stores_create_get_update_delete(),
            )
        )
        results.append(
            ("Vector Store Files list", self.test_vector_store_files_lifecycle())
        )
        results.append(
            ("Vector Store Search schema", self.test_vector_store_search_schema())
        )

        self.log("\n" + "=" * 60)
        self.log("Test Summary:")
        self.log("=" * 60)
        all_passed = True
        for name, passed in results:
            status = "PASSED" if passed else "FAILED"
            self.log(f"  {name}: {status}")
            if not passed:
                all_passed = False
        self.log("=" * 60)
        if all_passed:
            self.log("All OpenAI API validation tests PASSED.")
        else:
            self.log("Some tests FAILED.")
        self.log("=" * 60)
        return all_passed


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI API validation E2E (contract compatibility with upstream)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Unused when validating OpenAI API directly; kept for compatibility",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    test = OpenAIAPIValidationTest(args.base_url, args.verbose)
    success = test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
