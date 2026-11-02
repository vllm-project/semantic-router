"""Vela 3.0 d3 sizes (``vela3/3``): vela3-lite, -mini, -flash and -turbo.

Entries pin a revision, the SHA-256 of every file the family loads (the
adapters, the span heads, the configs, the router questions and the label
definition cache), the expected identity (which covers the base), the loaded
parameters (the base's and every text adapter and head) and the Decision 3.0
base each package's ``vela3_config.json`` pins (``base``). The repositories are
public; the pins are the revisions the records measured.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel


def _vela(
    name: str,
    revision: str,
    identity: str,
    parameters: int,
    base: tuple[str, str],
    memory: float,
    files: dict[str, str],
) -> BuiltinModel:
    return BuiltinModel(
        repo_id=f"{ORG}/{name}",
        revision=revision,
        family="vela3_d3",
        model_sha256=identity,
        manifest_sha256="",
        loaded_parameters=parameters,
        backbone="qwen3_5_text",
        min_device_memory_gib=memory,
        base=base,
        files=files,
    )


MODELS: tuple[BuiltinModel, ...] = (
    _vela(
        "vela3-lite",
        "c789afd6fbeaceb3c734b95b9ca3ad9772a7f434",
        "c0f959071e733c0642ca966c170cce8d228dce8ad58b546ff0f9c72302a61600",
        902_579_779,
        ("vllm-sr/d3-lite", "9c10703193afc7c7d28c88201c0364bfbbf58320"),
        4,
        {
            "config.json": "c6bf4eb4102e97a8464d8fabd364b5bf47e3cb9ee58f487d359207077bfae3d2",
            "decision_df_lora.safetensors": "2c156cfcd29823b9c6f4d23c4e6469d7b22ab78505a3d5174208d2b22c4e8453",
            "decision_fc_lora.safetensors": "071934ea85c7399f5bb8aacbdb35f4e7f602e5f5b3308216b38708f7c0dfa580",
            "decision_hz_lora.safetensors": "23f78fe3d43c2f9a130af91a47985bb7b1d3c12c1dfd8c1fe1da760c8b7f1185",
            "decision_lora.safetensors": "c01f1cacfe606cb9600459254c97f96a1e3e6356c4dec1f74fbbb36ace9f3535",
            "label_definitions.json": "4cd699d86c9d41a7947e19ae944541d9d552fc68e94b7c11d40b9b67b1afe62c",
            "ner_head.safetensors": "f25e2272da482a1214361d8d1ace58d3069d95ee859acb0c374f796b1c234c44",
            "ner_lora.safetensors": "aadcdbc0dc7bc5f464cdf6c8fa3671e7f1cc69b9b275373b88db186a86d31a8c",
            "relation_head.safetensors": "f1b64e1e2d32b37c27058911ac9df145540c08518c38543d2968deda95a7688e",
            "relation_lora.safetensors": "5352dc1d0ee6d859b82756baafc522c9dd0fa4f20ee5d2f3c31234327d1ab118",
            "router_questions.json": "e304705ed0bdf825cd1b88374fa80e1e3db26225857420226796f397d0ae5b60",
            "span_head.safetensors": "9abec87ef24a0b1b3101ed1a4245859186a581a4efed49c5ce2c7a945f64b14f",
            "span_lora.safetensors": "11cc6312d4e5862bb665d71d14aa4887499cc23a956e4069fc837afaca8116be",
            "vela3_config.json": "aa3508dfbf7942be09d35624ba64bd934bfa89513e10831cd141a43e3ebce96e",
        },
    ),
    _vela(
        "vela3-mini",
        "bbc0206efbbbee5d41955b429e98238b5ad32252",
        "dc5ddf7939e712b477a6075b61f84c4f23bcb932084c1ed8abe5b853f3411532",
        4_792_108_035,
        ("vllm-sr/d3-mini", "566da0ea398e27184472add700499245ed09af35"),
        16,
        {
            "config.json": "1859126a9061dd4f8eae29a79d8ebeb7a5b9ac7eb1cf7ca91422a433ee7547e6",
            "decision_df_lora.safetensors": "2455f8bd69170bde42b8fdc3511281829642c70df148e41755561a8cd0badbb4",
            "decision_lora.safetensors": "ebdd12efa205265ae756f3c3aeec13c8f98cb00b89fb348c3c66607f81bbaa7a",
            "decision_optin_lora.safetensors": "52992183539cf7ae5fbd6ce453615eef069fc76bd38c5b1f24d420c3e75132b0",
            "label_definitions.json": "4cd699d86c9d41a7947e19ae944541d9d552fc68e94b7c11d40b9b67b1afe62c",
            "ner_head.safetensors": "03dff7640e6fec1ad8ce8cd4af14e2a4e849615d7c296e931e80520a98cb4a39",
            "ner_lora.safetensors": "815a221cba6799ab104ac5725be73e9e8a6589354f0d6082b53ebd61a6dfe3fd",
            "relation_head.safetensors": "ae3d69387d4853cd0eb7dc80f84282946ee25f5b8eaa23e10af8ed9a2d5a4a1e",
            "relation_lora.safetensors": "45ca3ef715bbb4f35c5576915702004c9aad303251c24c6468eeb96ac65bce18",
            "router_questions.json": "8ed409d39cf03e49ccc5c5868643d73b72aa078497c848d214a370dd317fe3ec",
            "span_head.safetensors": "a06c85629cbe94c888bd9203cccafbaf1ede11f1864f9da4fc1ed4233c037d63",
            "span_lora.safetensors": "2ff18ab7c99a4a3c59bba4941e23c179d859690d62579de529828a07183746a7",
            "vela3_config.json": "7488b44a2eb22013b332335d71f7cf6d2091a543c2b4c9c13be11fa5750baa15",
        },
    ),
)
