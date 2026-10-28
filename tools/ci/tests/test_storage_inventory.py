"""Source annotations keep service tests out of hermetic execution without hiding cases."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_inventory import go_storage_tests


class StorageInventoryTests(unittest.TestCase):
    def test_source_annotation_discovers_exact_package_backend_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "src/semantic-router/pkg/store"
            package.mkdir(parents=True)
            (package / "redis_test.go").write_text(
                "// StorageIntegration: redis\nfunc TestPersistent(t *testing.T) {}\nfunc TestFixture(t *testing.T) {}\n"
            )
            self.assertEqual(
                go_storage_tests(root), {"./pkg/store": {"redis": ["TestPersistent"]}}
            )
            (package / "duplicate_test.go").write_text(
                "// StorageIntegration: milvus\nfunc TestPersistent(t *testing.T) {}\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                go_storage_tests(root)

    def test_replay_and_response_redis_are_in_required_service_inventory(self):
        inventory = go_storage_tests()
        for package, backend in (
            ("./pkg/routerreplay", "postgres"),
            ("./pkg/routerreplay/store", "postgres"),
            ("./pkg/vectorstore", "postgres"),
            ("./pkg/contextcompression", "redis"),
            ("./pkg/responsestore", "redis"),
        ):
            with self.subTest(package=package):
                self.assertTrue(inventory[package][backend])

    def test_exact_valkey_cases_are_required_service_tests(self):
        inventory = go_storage_tests()["./pkg/cache"]["valkey"]
        for name in (
            "TestValkeyCacheIntegration_ExactRoundTripAndPartitionIsolation",
            "TestValkeyCacheIntegration_ExactMaxAge",
            "TestValkeyCacheIntegration_LegacyStringOverwrite",
        ):
            with self.subTest(name=name):
                self.assertIn(name, inventory)

    def test_replay_pagination_is_required_alongside_metadata_migration(self):
        inventory = go_storage_tests()["./pkg/routerreplay/store"]["postgres"]
        self.assertCountEqual(
            inventory,
            [
                "TestPostgresMetadataIntegration",
                "TestPostgresQueryBeyondTenThousandIntegration",
            ],
        )
