"""Vela 2.0 (Phase 3): 0.3B, 0.8B, 4B and 9B.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_vela2.json`` and the decoders' (0.8B, 4B, 9B) kernel
choices in ``registry/kernel_choices.json``: those of the Decision 2.0 model
each one's backbone comes from (Eos-0.8B, Nox-4B, Lux-9B), the same tuning
keys. ``reduced`` names the copy
``max_speed`` may load per device class, where ``docs/records/vela2-parity.md``
and ``vela2-performance.md`` show it at the accuracy floor and faster. The
repositories are public; the pins are the revisions the records measured.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded


def _vela(
    size: str,
    revision: str,
    identity: str,
    manifest: str,
    parameters: int,
    backbone: str,
    memory: float,
    files: dict[str, str],
    reduced: dict[str, str] | None = None,
) -> BuiltinModel:
    return BuiltinModel(
        repo_id=f"{ORG}/Vela-2.0-{size}",
        revision=revision,
        family="vela2",
        model_sha256=identity,
        manifest_sha256=manifest,
        loaded_parameters=parameters,
        backbone=backbone,
        min_device_memory_gib=memory,
        files=files,
        reduced=reduced or {},
    )


MODELS: tuple[BuiltinModel, ...] = (
    _vela(
        "0.3B",
        "a3209a50dc3ebd7e3b7520440d8fba666000f4c4",
        "7e786f91413961d76a8b85318f70fea1509124fe33709a8c12d6badf8d2ccda4",
        "",
        309_114_371,
        "modernbert",
        4,
        {
            "calibration.json": "f8557b3922e9bef33e688ece12da20436b5ae868ed7ff1375b346bd26403506a",
            "config.json": "81e08c8f8f2a4a6205f05b67d5682e42fc5249cc11e5231c456929ae0534d675",
            "model.safetensors": "bd37dd0db177d766bc554eded78ae8493bc248172d0480f6a543b30932a0cc8e",
            "tokenizer.json": "e4b670c5ab72158f35de150792133e8c92adb4ba677d26b99d22c77f34c63faf",
        },
        reduced={"cpu": "float32-packed"},
    ),
    _vela(
        "0.8B",
        "a778eb2ae2304cfa72fca7e53a19136dea5be012",
        "e231da382b5d3d45f253e9b3ff12afe7044b56d95afa4c2c0ec412c036024949",
        "a8fb5b188593463209475a7b51ceb97531b58bb1d7445b8b6a2e1a6333fd1ae2",
        755_685_699,
        "qwen3_5_text",
        4,
        {
            "MODEL_MANIFEST.json": "a8fb5b188593463209475a7b51ceb97531b58bb1d7445b8b6a2e1a6333fd1ae2",
            "broad_head.safetensors": "687d7a435d6de846796274e417b6333f83df3085a6e7c3bca4c01cb4d75a30f0",
            "calibration.json": "d96e447962d0dc1d58fa8422dddbae8b99960e21fd75a10ef9b096f700a211f5",
            "config.json": "0637ca193b9159b6ece25837913956477164d3392d9b5dd1e6b74b95a484004a",
            "model-00001-of-00001.safetensors": "c70237ecd579c60289c177fa9806e71c24840c2fc557b5c76af1395dc3081fa6",
            "model.safetensors.index.json": "b53d1c79219677289bdb1541a1373dfe53119884f248b23c94a27aef7338919d",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        },
    ),
    _vela(
        "4B",
        "c1e64d4f872cb38bc58502e6888340100bab9d55",
        "c47ee0594199d66f924c81c1664cb97a9f2cfaeb12d484c232b3d6abe67ec07b",
        "3e5a26db6dd50cc3939502ec861cdc0e5f6a5de8947815d598d3aea6f4624a0c",
        4_213_980_675,
        "qwen3_5_text",
        16,
        {
            "MODEL_MANIFEST.json": "3e5a26db6dd50cc3939502ec861cdc0e5f6a5de8947815d598d3aea6f4624a0c",
            "broad_head.safetensors": "45dd5eafa620a5b4d3e060eaf8baadbfeaab4d3086dc3a63a6734068e441e20c",
            "calibration.json": "2a3f3cfa2d1cf6fa133d887960261b71c09420899f817f4156677962ac02460f",
            "config.json": "c0318849c586fe42f98db289c9d959642a8e2020726f2b3cb4685ccfc9912c40",
            "model-00001-of-00003.safetensors": "32a4c236433eb321eed9c8f1b3c538b1e92c9d10434f881f19297ec31df8a3f2",
            "model-00002-of-00003.safetensors": "f632598bf50da7f07fca7b6a0588c2b528c05158d03877205450d36dc3ce9662",
            "model-00003-of-00003.safetensors": "6f59ab795dc73e0fdb5f25cb0522808537097db2b9b0e147fc4c33f12ff90a80",
            "model.safetensors.index.json": "24e5eea6b54282cdf32a7f8d746f9c7b3f8c2ef66bcd57b333fd0022b7c74ca0",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        },
    ),
    _vela(
        "9B",
        "bc8761637d8788619dfbaf6d8890128efe85fd40",
        "76ac16a7231b02989465212389dc55d42a45655c4e574599be3abe35c0e56cd5",
        "8ccd2ccb2f91f7b4ee99df2e3bf885cb901f2567cfb0a6b6093778ba14dfc05f",
        7_949_850_627,
        "qwen3_5_text",
        24,
        {
            "MODEL_MANIFEST.json": "8ccd2ccb2f91f7b4ee99df2e3bf885cb901f2567cfb0a6b6093778ba14dfc05f",
            "broad_head.safetensors": "c376f1668247ff902947d1ffd69c92195c7b74116dedd958aebd31ff5f1b580c",
            "calibration.json": "9fa0f31ce1c7b784220fd73320ee64da9b3b789c403755af91bcbefb5740d766",
            "config.json": "4770113c223c399500382f07746cfe384465409e16f07b2cea49ced9f2547aa3",
            "model-00001-of-00005.safetensors": "444ef8c7f74c5e52639ae39cb692b5f93f9df419f58c1d1ca048046496e702b2",
            "model-00002-of-00005.safetensors": "6f11c43ba2f04be71a8102eaa1fce2b73ef9c9afe7cb5c6e6690bdf941427cc7",
            "model-00003-of-00005.safetensors": "dc81308a51d4357a7483701a061ebbeee9a90f7fd3a0716a99ff0d58a6330b9d",
            "model-00004-of-00005.safetensors": "d08cb842d293fedb05308c299624cd3459c2007ec2b7ad75267f67cb225ace70",
            "model-00005-of-00005.safetensors": "1aec88bbd34d725ac0a9c152bc1d10d2db25bca1f9186f5449d7429a4b034a81",
            "model.safetensors.index.json": "b8c6ef31bad3b3f55952261553827df322e666e2393f85856e435af70e393c4e",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        },
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_vela2.json", "answers", "golden_answers")
MODELS = with_recorded(MODELS, "kernel_choices.json", "choices", "kernel_choices")
