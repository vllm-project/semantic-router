"""Vela 3.0: vela3-fast (the UltraBERT-380M encoder and its adapters).

Entries pin a revision, the SHA-256 of every file the family loads (the
backbone, every adapter it routes to, the configs, the router questions and
the tokenizer), the expected identity and parameter count; references live
in ``registry/golden_answers_vela3.json``. The repositories are public; the
pins are the revisions the records measured.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded


def _vela(
    name: str,
    revision: str,
    identity: str,
    parameters: int,
    backbone: str,
    memory: float,
    files: dict[str, str],
) -> BuiltinModel:
    return BuiltinModel(
        repo_id=f"{ORG}/{name}",
        revision=revision,
        family="vela3",
        model_sha256=identity,
        manifest_sha256="",
        loaded_parameters=parameters,
        backbone=backbone,
        min_device_memory_gib=memory,
        files=files,
    )


MODELS: tuple[BuiltinModel, ...] = (
    _vela(
        "vela3-fast",
        "7cbefd4b92b57234bd0ffb9872451e81abcdac79",
        "5b6858010269f2ce83ee59fdb94b920fac550741970dc045b4b479b86bb6cf92",
        1_518_582_809,
        "ultrabert",
        12,
        {
            "config.json": "a85b358ac11b71d780deb09e8a082efddc67938a2eb4fc16d68d8ae888c69d14",
            "domain_adapter.safetensors": "1aa3f1fdf590a369ba23ebf2b2b5896a0673a007b19597cfc6e7ca3a77cb067c",
            "evidence_adapter.safetensors": "8aeda02d53ca0f29b1888c567b0e7199dd15ceea0b77faa26448f293840ac774",
            "factcheck_adapter.safetensors": "fd90e966d466177e4bd2db7701372a98990d34dc9eccc9ec871113c929039b3b",
            "feedback_df_adapter.safetensors": "c1d7e153fa6be7cd9aa704cd57077596d23949ea0beb9d6650cc3216021cffbb",
            "model.safetensors": "2294745881d9a01b7d049133b57e5fed2aec6ae1d34ef1ab9fa16f047fc35c8e",
            "ner_adapter.safetensors": "387f409c7f954e696c2d95f6b85829e84e6ea8ad52128f2cf1340f7ee7045f8a",
            "relation_adapter.safetensors": "a448606d98c5f0c5e0e3249ac00465c69f58fa6995555b344e6c1d89b99c4d46",
            "router_adapter.safetensors": "30c7f350bcc4060b302b26f57d73c965da8e8b75509c1262ae953942b11adac8",
            "router_questions.json": "5126f841279211df70ab027b6c124187dd444e22dbd10202f4e17de96ad432d5",
            "span_adapter.safetensors": "15eb0f11e4261bc6f54e4a6a6cc15951575a510dd1d0bd598e6a6d7a15800fd8",
            "span_sets_adapter.safetensors": "73c904acf7a765a38fd793b0d5d9d02633f9c5938620c9e063063aa1458cf006",
            "tokenizer.json": "38734bb17ef19f84c67b5eb5692bea19801ffd02cb5c5ffc11429e76b0242e08",
            "vela3_config.json": "ee5088605aa5151210fbc12e3f6ee0c9a382dddf90d3a43ac606c0df305d9ff2",
        },
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_vela3.json", "answers", "golden_answers")
