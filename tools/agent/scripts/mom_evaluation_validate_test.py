import importlib
import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

mom_evaluation_validate = importlib.import_module("mom_evaluation_validate")


class MomEvaluationValidateTest(unittest.TestCase):
    def test_manifest_schema_accepts_mom_v1_manifest(self) -> None:
        manifest_path = (
            REPO_ROOT / "config/recipes/built-in/latest/mom-v1/mom-evaluation.yaml"
        )
        schema = json.loads(mom_evaluation_validate.MANIFEST_SCHEMA.read_text())
        errors = mom_evaluation_validate.validate_manifest_document(
            mom_evaluation_validate.load_yaml(manifest_path),
            schema,
            manifest_path,
        )
        self.assertEqual(errors, [])

    def test_result_schema_accepts_blend_scorecard(self) -> None:
        result_path = (
            REPO_ROOT
            / "config/evaluation/scorecards/mom-v1/mom-v1-blend/1.0.0/mom_eval_result.json"
        )
        schema = json.loads(mom_evaluation_validate.RESULT_SCHEMA.read_text())
        errors = mom_evaluation_validate.validate_result_document(
            mom_evaluation_validate.load_json(result_path),
            schema,
            result_path,
        )
        self.assertEqual(errors, [])

    def test_all_mom_recipes_validate(self) -> None:
        schema = json.loads(mom_evaluation_validate.MANIFEST_SCHEMA.read_text())
        for manifest_path in mom_evaluation_validate.find_mom_recipes():
            errors = mom_evaluation_validate.validate_manifest_document(
                mom_evaluation_validate.load_yaml(manifest_path),
                schema,
                manifest_path,
            )
            self.assertEqual(errors, [], msg=str(manifest_path))


if __name__ == "__main__":
    unittest.main()
