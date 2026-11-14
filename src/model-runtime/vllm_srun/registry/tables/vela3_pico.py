"""Vela 3.1 picos: vela3.1-pico-380m and -150m (one UltraBERT encoder each, no adapters).

Entries pin the ``v1.0`` tag commit of each repository, the SHA-256 of every
file the family loads (the weights, the two configs, the router questions and
the tokenizer), the expected identity (the weight file's digest), the loaded
parameter count and a device memory floor of about twice what a GPU holds
once the model is loaded (1.42 GiB and 0.64 GiB on an MI325X); references live in
``registry/golden_answers_vela3_pico.json``. The repositories are public; the
pins are the revisions the parity record measured
(``docs/records/vela31-pico-parity.md``). The bundled Python and the int8
ONNX encoder are not pinned: the family never reads them. ``SHA256SUMS`` is not
pinned either; a pinned model is verified against the pins, and the family
reads that file only when the package has one (required when nothing is pinned).
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded

COMMON_FILES = {
    "router_questions.json": "e3eced909d4be3096e7c7cd7301790deb9f362e78e27b71cde45b92852edcd37",
    "tokenizer.json": "38734bb17ef19f84c67b5eb5692bea19801ffd02cb5c5ffc11429e76b0242e08",
}


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
        family="vela3_pico",
        model_sha256=identity,
        manifest_sha256="",
        loaded_parameters=parameters,
        backbone=backbone,
        min_device_memory_gib=memory,
        files={**COMMON_FILES, **files},
    )


MODELS: tuple[BuiltinModel, ...] = (
    _vela(
        "vela3.1-pico-380m",
        "a79cbf66ee09245ef8f40f66afa9bcbd57ccf5a5",
        "82af9e582c58c2bc63fe5d3c556adf9b390ffc0224e50830c6693abab02577bc",
        380_961_027,
        "ultrabert",
        3,
        {
            "config.json": "5e0ee2866f57e0058afb5ceebdf8f4803515a6fe5a5a66acc94ff5245906c391",
            "model.safetensors": "65309b558d98eee844f496950f34688cb210aaa71a5401d4d5c0e37f6d7f8089",
            "vela3_config.json": "e16d2c299106d82b6934cdd0e58368625f08a1669474eeaca6f792b0fe068b0e",
        },
    ),
    _vela(
        "vela3.1-pico-150m",
        "dad77ad7a83c884b3e608a58c13cf25beba00803",
        "d3421c0e7d3510bb438179369402a6e1c6ada6bdab7aa217e630e4dee380c268",
        153_207_299,
        "ultrabert",
        2,
        {
            "config.json": "f10ac925e77c77a7d8214a0006c743477a47f718838961217066082a9b250938",
            "model.safetensors": "f0b4dd30a096d74219314912e632f4a59f6a4bc513364d8f8a40afbfe4826c38",
            "vela3_config.json": "a4706d50eaf151c936c54c63ed177f3f0a18960fe9438af39a0021c566d508bb",
        },
    ),
)
MODELS = with_recorded(
    MODELS, "golden_answers_vela3_pico.json", "answers", "golden_answers"
)
