"""Vela 3.0: vela3-fast (the UltraBERT-380M encoder and its adapters) and vela3-fast-vision (the UltraVBERT image package).

Entries pin a revision, the SHA-256 of every file the family loads (the
backbone, every adapter it routes to, the configs, the router questions and
the tokenizer), the expected identity and parameter count; references live
in ``registry/golden_answers_vela3.json``. The repositories are public; the
pins are the revisions the records measured.

``vela3-fast-vision`` pins the ``vela3v-encoder/1`` image package: three
UltraVBERT backbones (the raw image-hallucination/OCR backbone in
``model.safetensors``, the decision-trained ``backbone_h2`` and
``backbone_v2`` two tasks name as their ``backbone_file``) and the span,
decision, box and OCR task adapters. The identity is the digest set of
those seven weight files. The parameter count sums their safetensors
headers and is provisional: the serving path's load count deduplicates the
backbone tensors the span and box adapters overwrite, as ``vela3-fast``
shows (1,518,582,809 loaded against 1,504,140,559 by headers). The
``text/`` subtree is a nested text-encoder package with six more weight
files and is not part of this pin; ``processor_config.json`` stays unpinned
until the image serving path loads it.
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
    _vela(
        "vela3-fast-vision",
        "1e13154c1c2137654c79c4a08b912f0085bf9fe8",
        "bc533cd701b09cace4b44215831ff037049c4d25fe08bf3c1c7db8776508cc5c",
        874_361_092,
        "ultravbert",
        6,
        {
            "backbone_h2.safetensors": "65e0c0284eb8c6e494c058d1988ad1b3d2eab60c368bb3d704e09174d1220bd5",
            "backbone_v2.safetensors": "46e65f6e6b906a64e4ad57e2e76957522bd01e2eeb5d8467078b895410d1e0aa",
            "box_adapter.safetensors": "62d05d58ddbb7f9248dff4e84dc4864c09c78a56e19088cae112ccbd9035fb6b",
            "config.json": "96c0a13f682092cb4b7289ec10d54179dd0085bbc988cba269a7b3d1ace1c079",
            "decision_adapter.safetensors": "8a8ddc13ccc09ebe41f4200b8685a72a7c19d4567c7c1a85f600b25dad53ae55",
            "model.safetensors": "9387cfd5bc53082b6ff634453c6b0cb3123a67e4c197209e308ecfaf7a578900",
            "ocr_adapter.safetensors": "882f2d71d6aee9d90f75b68d35c6eb74388fa96c1013d959b0ab18fe8b0f9719",
            "router_questions.json": "f7bf64a8c5265a843372147aef70a8babaa584ea9b9e2e7751fa1138d656b10f",
            "span_adapter.safetensors": "9cbfd04d5960818001c1b6ff7a3f02a5595dd73f3c5d32a256fbe1855f4055bd",
            "tokenizer.json": "bda9af747539d9444eca3fc9df922811ae318d93ced0b83ecf2001c77a925e8a",
            "vela3v_config.json": "f8d159468b49ee268c29d29c99ec488226ef7524fd967caa34d8d71f333f71f2",
        },
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_vela3.json", "answers", "golden_answers")
