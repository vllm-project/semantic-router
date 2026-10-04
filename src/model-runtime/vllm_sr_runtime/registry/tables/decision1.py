"""Decision 1.0 (Phase 2): Kai, Lex and Route on the Vela encoder; Eos, Sol, Nox and Lux on Qwen3.5.

Entries pin a revision, the SHA-256 of every file the family loads, the
expected identity and parameter count; references live in
``registry/golden_answers_decision1.json``. The packages' bundled runtime
tunes FLA's kernels in every process, so the Qwen3.5 sizes run with the
pinned kernel choices of the Decision 2.0 package of the same architecture
(identical layer widths, so identical tuning keys); the parity record runs the
bundled runtime with the same pins.
"""

from __future__ import annotations

from dataclasses import replace

from .common import ORG, BuiltinModel, with_recorded
from .decision2 import DECISION2_MODELS

MODELS: tuple[BuiltinModel, ...] = (
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Kai-0.6B",
        revision="79263ba4c4befac3845e7c1111c2c679d0716623",
        family="decision1",
        model_sha256="90bc61362eb40a7d78e3695afe0fbe0a24f8e152515fa1adb1c4da814f045885",
        manifest_sha256="",
        loaded_parameters=571_909_635,
        backbone="modernbert",
        min_device_memory_gib=4,
        files={
            "config.json": "43357bfd40be87773266494f479530db88de7f4ff1c73132f01ff86bd325b870",
            "native/choice_encoder.safetensors": "bb520f1e36035862b3f9b564a24647e46704b8993d8d2f8659b3470252299792",
            "native/decision_config.json": "f9b697bd57d0ae42041ca0a075f05c6d0123bd58adbd7e096ef25d0b55cf5d99",
            "native/decision_heads.safetensors": "730fa21bac15e7ba7e239658ef521aa74948dd7d0a9dc417c4aef7d36c084267",
            "native/encoder/config.json": "7aff915e9f159305e0bef3eb0206416f99b8560260b8969b35f1dcd54aaad1a5",
            "native/encoder/model.safetensors": "4c95e05a3abf24ec93ce209e22915c1e3f2199b4c9d9c76e88f27dee6a2123c3",
            "native/score_encoder.safetensors": "400d1d9cd29452b50250a96ab121df31ca40c5cfbabe94ebe634ab96aec79678",
            "native/tokenizer/special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "native/tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "native/tokenizer/tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
        reduced={"cpu": "float32-packed"},
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Lex-0.6B",
        revision="a5ba6895347218eb8acd818420001b59a00b64c2",
        family="decision1",
        model_sha256="60e1329eb96d79eb139f2057e52f1fb7edb5999ec4abf8c1c746a93734612c2c",
        manifest_sha256="",
        loaded_parameters=571_909_635,
        backbone="modernbert",
        min_device_memory_gib=4,
        files={
            "config.json": "77ecf8ffb689f74ab7686570aedeeefef4bc9f739d8d5e09f77358b425965dcc",
            "native/choice_encoder.safetensors": "9516cc841c485c98b27b4f63d2ea8e604fe8064121173064b113da8a6bf57ef6",
            "native/decision_config.json": "e157e272f1f4102053881817d6b28053919b7a7ec56fd8b9da506ad8c7a095d9",
            "native/decision_heads.safetensors": "bce3ee658a978a19c48c605b921cff994f892e7fc17abd6f8645f07428fbc36f",
            "native/encoder/config.json": "7aff915e9f159305e0bef3eb0206416f99b8560260b8969b35f1dcd54aaad1a5",
            "native/encoder/model.safetensors": "daaafd81c4ed767d203e24226d78e792800068a7dac344aa043844b82a6306c7",
            "native/score_encoder.safetensors": "4f45795977846ef4c31b34e4f35bf95dabf5951d71358bab32c9a94814a84a53",
            "native/tokenizer/special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "native/tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "native/tokenizer/tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
        reduced={"cpu": "float32-packed"},
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Route-0.6B",
        revision="deed1f29dad146d2138588025afb83dbdaf3c133",
        family="decision1",
        model_sha256="e4a6852defe9fbc0ef5dcd7d2b4e0418188d43d1eab779967b25701d0af80fad",
        manifest_sha256="d7c4b0d3885f7e505c247ff40749abb193c923828713138fdaa3c9201db2da0c",
        loaded_parameters=571_909_635,
        backbone="modernbert",
        min_device_memory_gib=4,
        files={
            "MANIFEST.json": "d7c4b0d3885f7e505c247ff40749abb193c923828713138fdaa3c9201db2da0c",
            "QUESTIONS.json": "7d97837c36dfdfb425e97e2bf0ebac34caf4950505a6b931c2b49e6f7f2be7e2",
            "config.json": "ed67600f1f0a098a07596440f79e403ccfb9954a837683529980192c75ba51f2",
            "native/choice_encoder.safetensors": "676e37a278a567bea8d5912c346a5395d69a9c2b9b980e047425f7a28f1def98",
            "native/decision_config.json": "87a9d42e2b0df9a495d6150272382d9d33b2d5d9e15d8e9f34d25c6d5824fddf",
            "native/decision_heads.safetensors": "eddb616c8971c2529cecfd51f3757e69fb794c68d1a04566e9304acba6c3a30b",
            "native/encoder/config.json": "7aff915e9f159305e0bef3eb0206416f99b8560260b8969b35f1dcd54aaad1a5",
            "native/encoder/model.safetensors": "c0d26abb1a232db53c155e125720b22e9dc759a9105d9541e1ca453d7523fe6d",
            "native/score_encoder.safetensors": "400d1d9cd29452b50250a96ab121df31ca40c5cfbabe94ebe634ab96aec79678",
            "native/tokenizer/special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "native/tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "native/tokenizer/tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
        reduced={"cpu": "float32-packed"},
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Eos-0.8B",
        revision="2ca39a23e3f4500873ebcd8f97394268e27d3167",
        family="decision1",
        model_sha256="2ac899236318a62c48dbade9e088146eaf727f1047f7ef93c198b4a96fe5457c",
        manifest_sha256="",
        loaded_parameters=753_446_208,
        backbone="qwen3_5_text",
        min_device_memory_gib=4,
        files={
            "backbone/config.json": "5cece452e606f8ffca1790b79018fed5b0fde7d1a0791d375ca09492e0d1809b",
            "backbone/model.safetensors": "613f491da2794c5d22f68e81bd2d41000c5675bc4538180148e6e453e8198abd",
            "config.json": "a957a2a071ca4ded5de2a26eff42f76dfa043143d3cb885b5b4338e69d1c2384",
            "decision_config.json": "71fde3c2ea6f4bf4e781028d7f8f39da73599f9d868979efbfbd8b7f43dd68c3",
            "decision_head.safetensors": "75f4ee8b512aea8f74fdcb42849b4cd37cfb37ff17c5d0b5fda1bb7b4f6d72f6",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Sol-2B",
        revision="5c698b1ad76abf7c35fc4957519c47668f54d23a",
        family="decision1",
        model_sha256="ccbd5382506e529a63cd2fbacd5a62351dfb5f8744912ca671fea24af4cd283a",
        manifest_sha256="",
        loaded_parameters=1_883_930_944,
        backbone="qwen3_5_text",
        min_device_memory_gib=8,
        files={
            "backbone/config.json": "e7bed6f1a1a4d5f029a28040e8687a122b323798d4674666443f6d78e9475a05",
            "backbone/model.safetensors": "99f97c2564acc1d43b89a1c547b63ce1d87f062637d0a8ed41eb40d752849e36",
            "config.json": "76e163bff7ff572666ee6e261f309ca52f01677e9a8c9b9def6f19fc0ae149c0",
            "decision_config.json": "535a7aa88ca6693f2e6fc2c5721166173931e408eaad3c18dd1c84228cdeeb13",
            "decision_head.safetensors": "5c3dce45115fba20192c99a32fa89a379c2a6617f02e3647dc8f05e48dcaf215",
            "temperature.json": "f0cbe7323441ceaf6abc8dd3f9b1832f5d4121a803f18e9643406aadd53c6efc",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Nox-4B",
        revision="7f65e1db2c55a11ab8b558a8027c8883752f192d",
        family="decision1",
        model_sha256="694b14491c226ceef873d778fe254fca6105f2ed84b2c5c9db41e255f084ca16",
        manifest_sha256="",
        loaded_parameters=4_208_383_488,
        backbone="qwen3_5_text",
        min_device_memory_gib=16,
        files={
            "backbone/config.json": "ae3a463b32e95b6cc207a7af4f1defb4195f388eb6f9ff19d2690b73d4966953",
            "backbone/model-00001-of-00003.safetensors": "5ebccc395ef9fa4d61c79894ecefcdaa2a319c1c154221e1ad73663b6f82aacc",
            "backbone/model-00002-of-00003.safetensors": "990e2e79fc1ab009df9846ef9fddb31f4b988589bbda84f4728dbaa4196d7fcd",
            "backbone/model-00003-of-00003.safetensors": "c47859a3192bdae5003e4732fb911c8dddcf6e8caa167b4971e3a9801b828d64",
            "backbone/model.safetensors.index.json": "1602d52e38d81586af85bc4ce29ce082c5fc5877c763b1ebcab7545320016599",
            "config.json": "58831881a885b58aa995575812bdb1d9bae9f870038323580ddc887f5b2eca74",
            "decision_config.json": "207b345c1ac03f1836effb48f6d2e13c0a979f8cfafee389877a4237b6eae980",
            "decision_head.safetensors": "9cb6f639714e31bcb76b58eaf94af0b72ac3db9d091ebcda45d9b575efd489de",
            "temperature.json": "69d80e5b215e2c1e6872f146fdb7ea5aa95c9fd678ef96208960d50e4e7b635d",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/Decision-1.0-Lux-9B",
        revision="2064c84daf599447f840719ef8e86db152a56851",
        family="decision1",
        model_sha256="364767e22dd6bcd5172da76c020c2422fb6de97153563c465bd067bfae4d170e",
        manifest_sha256="",
        loaded_parameters=7_940_895_744,
        backbone="qwen3_5_text",
        min_device_memory_gib=24,
        files={
            "backbone/config.json": "4a87e4e7e11a11284a210066648b6d7616fda2872caeadec015aecb26dd9cffe",
            "backbone/model-00001-of-00004.safetensors": "38b8c6b3cacebb8e5e24557fb29854efd0db217224ed0c5fb7482513de7436ca",
            "backbone/model-00002-of-00004.safetensors": "2b5686373f785b0043289bcd0d374bfdccd8b513754b66401ef54bd4c8ad2403",
            "backbone/model-00003-of-00004.safetensors": "ff152c4b46e7f2df8e257c7dc2818d569fd4ab0dc659494b26d65004f0f83d50",
            "backbone/model-00004-of-00004.safetensors": "99d7d3bf53cf1a3c5c355bdf7a139c7daa3ab47ce703980eaa8a27106590a8a6",
            "backbone/model.safetensors.index.json": "f943816d8882f0acb572029805817240c2310d0d7e5b76fe1ececff63eab0686",
            "config.json": "fe8780776f5bd30e3cd6a34e311b3168249d9a039d35f92d614a07fddfad848a",
            "decision_config.json": "20061ac6d00d988e3d2fa44b820b5501fb44ccb22053876994fc6645b532409f",
            "decision_head.safetensors": "f810788b27ee5fbeede232041d2b61022033d2b717d1d90070d2d903fca6137f",
            "temperature.json": "ca7dfe0c3f28ab5d66804688ad779e2bc1f98f2e57a97669033a9d3bc4bf2e63",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
        },
    ),
)
MODELS = with_recorded(
    MODELS, "golden_answers_decision1.json", "answers", "golden_answers"
)
_SAME_ARCHITECTURE = {
    model.repo_id.replace("Decision-2.0-", "Decision-1.0-"): model
    for model in DECISION2_MODELS
}
MODELS = tuple(
    (
        replace(model, kernel_choices=_SAME_ARCHITECTURE[model.repo_id].kernel_choices)
        if model.backbone == "qwen3_5_text"
        else model
    )
    for model in MODELS
)
