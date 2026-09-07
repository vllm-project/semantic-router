"""Import-light contracts for the split mmBERT-32K trainer modules."""

from __future__ import annotations

import ast
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

from src.training.model_embeddings.mmbert_32k.config import arguments_to_argv
from src.training.model_embeddings.mmbert_32k.embedder_cli import (
    build_parser as build_embedder_parser,
)
from src.training.model_embeddings.mmbert_32k.foundation_cli import (
    build_parser as build_foundation_parser,
)
from src.training.model_embeddings.mmbert_32k.foundation_data_cli import (
    _split_optional,
    _split_optional_ints,
)
from src.training.model_embeddings.mmbert_32k.foundation_data_cli import (
    build_parser as build_foundation_data_parser,
)
from src.training.model_embeddings.mmbert_32k.reranker_cli import (
    build_parser as build_reranker_parser,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"

_PUBLIC_NAMES = {
    "embedder.py": {
        "SelfDistillationLoss",
        "convert_bge_to_triplets",
        "create_evaluator",
        "get_batch_size_for_length",
        "get_model_info",
        "load_bge_data_directory",
        "load_bge_jsonl_file",
        "load_evaluation_data",
        "parse_args",
        "test_layer_reduction",
        "train",
    },
    "foundation.py": {
        "ANCHOR_MASK_PROBABILITY",
        "EWCRegularizer",
        "RetrievalMaskingCollator",
        "StandardMLMCollator",
        "load_dataset_from_path",
        "main",
        "train",
    },
    "foundation_data.py": {
        "concatenate_to_long_context",
        "download_from_huggingface",
        "main",
        "retokenize_dataset",
        "tokenize_dataset",
        "tokenize_local_files",
        "verify_dataset",
    },
    "reranker.py": {
        "Matryoshka2DLoss",
        "Matryoshka2DReranker",
        "RerankerDataset",
        "RerankerExample",
        "collate_fn",
        "evaluate_model",
        "main",
        "train",
    },
}
_FORBIDDEN_ORIGIN_MARKERS = (
    "upstream blob:",
    "source repository:",
    "source location:",
    "code provenance:",
)


def _config(name: str) -> dict:
    return json.loads((CONFIG_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _module_all(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{path} must declare __all__")


class ModuleBoundaryContractTest(unittest.TestCase):
    def test_compatibility_entrypoints_reexport_historical_names(self) -> None:
        for filename, expected in _PUBLIC_NAMES.items():
            with self.subTest(filename=filename):
                self.assertEqual(_module_all(ROOT / filename), expected)

    def test_split_sources_do_not_restore_broad_ruff_suppression(self) -> None:
        for path in ROOT.glob("*.py"):
            with self.subTest(path=path.name):
                self.assertNotIn("ruff: noqa", path.read_text(encoding="utf-8"))

    def test_sources_do_not_embed_origin_metadata(self) -> None:
        for path in ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.relative_to(ROOT)):
                for marker in _FORBIDDEN_ORIGIN_MARKERS:
                    self.assertNotIn(marker, source)

    def test_pinned_config_arguments_remain_accepted_by_family_parsers(self) -> None:
        cases = [
            (
                build_foundation_data_parser,
                _config("foundation")["prepare_arguments"],
            ),
            (build_foundation_parser, _config("foundation")["train_arguments"]),
            (build_embedder_parser, _config("embedder")["train_arguments"]),
            (build_reranker_parser, _config("reranker")["train_arguments"]),
        ]
        serialized = json.dumps([arguments for _, arguments in cases])
        environment = {
            name: f"/tmp/{name.lower()}"
            for name in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", serialized)
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            for parser_factory, arguments in cases:
                with self.subTest(parser=parser_factory.__module__):
                    argv = arguments_to_argv(arguments)
                    namespace = parser_factory().parse_args(argv)
                    self.assertTrue(vars(namespace))

    def test_comma_separated_receipt_inputs_are_stable(self) -> None:
        self.assertEqual(_split_optional("en, zh-Hans,de"), ["en", "zh-Hans", "de"])
        self.assertEqual(_split_optional_ints("1, 2,3"), [1, 2, 3])
        self.assertIsNone(_split_optional(None))
        self.assertIsNone(_split_optional_ints(None))


if __name__ == "__main__":
    unittest.main()
