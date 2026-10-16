"""Vela 1.0 encoder task models (Phase 3) at the revisions the router pins.

Domain, Guard, Safety, Shield, FactCheck, Feedback, Modality, Hazard, PII and
Halu: one FP32 ModernBERT task checkpoint each; Embedding and Reranker: the
ModernBERT embedder and pair scorer with their ONNX exit graphs; and
Qwen3-Embedding-0.6B, the Qwen3 embedder the router has served. Entries pin
the revision, the SHA-256 of every file the ``task_heads`` family loads (and
so downloads), the identity those digests define and the checkpoint's
parameter count; references live in ``registry/golden_answers_vela1.json``.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded


def _vela(
    name: str, revision: str, identity: str, parameters: int, files: dict[str, str]
) -> BuiltinModel:
    return BuiltinModel(
        repo_id=f"{ORG}/Vela-1.0-Encoder-307M-{name}",
        revision=revision,
        family="task_heads",
        model_sha256=identity,
        manifest_sha256="",
        loaded_parameters=parameters,
        backbone="modernbert",
        min_device_memory_gib=4,
        files=files,
    )


MODELS: tuple[BuiltinModel, ...] = (
    _vela(
        "Domain",
        "f6354f54adcf38770f635ad903be2b00577f6c11",
        "ce836f431e224e2c8134b5b8be60236bccaf5eb900c848701bf99c65a950f826",
        307_541_006,
        {
            "config.json": "2741d5ac33b7c56fd3c716e6b434e4616d6f0d943b9f0634fbf3bb1e7dd8cafd",
            "model.safetensors": "be89c0202dc630796e0487b2cb661103782fc2d8aa181a8f92bd67b697d089a4",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Guard",
        "087f9e401012df839c83717b746967ac7aebfa3e",
        "5964044e44330e637ad28416ac2d4ef462fe5c988a0f8d65d2df2f984aa298fd",
        307_531_778,
        {
            "config.json": "fb5e4c914262269aa4c9d6f27901cd307ea802c985be4e02270af315f9ff2e17",
            "model.safetensors": "2d5c574b95c21450747311362ac95c6868e44b8d4f486aaab64d4cc6f408fd55",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Safety",
        "6e70e725a5f4d86da10f5be5e4dfd1da0358bb85",
        "d655ebdf2fea37a2efba428d891740e6af3e7b94ee475d07b3ccd91f59ab9a2f",
        307_531_778,
        {
            "config.json": "ace5f335464763f3314892e4df5446fd60d0265803dfe27b3f91bbd64f197ff8",
            "model.safetensors": "75523d448d0fd6a33cc2377967ef09f8798b3a52e789de6c569c89f9efc51050",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "22fd4a60565f25fee8ebd2866aab07456dd46cbbeb45eaf506529ed8451ee58f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Shield",
        "a981a99eeb05a2859b88b5cee9af4352897ec4ec",
        "ac4667a3903e8092888d7c68aa476ee678f617f3466c495718cc2d73e956d570",
        307_531_778,
        {
            "config.json": "ace5f335464763f3314892e4df5446fd60d0265803dfe27b3f91bbd64f197ff8",
            "model.safetensors": "ab3b50dfa0e8d646e70813ed13cad2739d4a5a8d0cc84ed18889fcf802d25755",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "3eb3326c920a4d2d8c4ceadb85f8fddd6e4b1c953eed79366817fd169f905378",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "FactCheck",
        "99ede1aba1563e59e416f744d25b3f6b7e9d8274",
        "c9db80f6c737e858162b8e0b59d25416c00f9b5a4c75ea53cbfcec021f3d82ff",
        307_531_778,
        {
            "config.json": "626b677d5ec948f26aab5f401eecc634f9c9b91a2bebc24c2bdd1e3a74d619a7",
            "model.safetensors": "5c946063aab3bbe1defcb0e7697d4275e652c1dc74f1b4d6789d681fd9899739",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "22fd4a60565f25fee8ebd2866aab07456dd46cbbeb45eaf506529ed8451ee58f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Feedback",
        "47434a7fd7c245c0c7c17564a000b3c56ccfec41",
        "554230a54eb4a9513bbcf049dd44ca1bfb2f9c814b0712f1177a97dee0d86ac9",
        307_534_085,
        {
            "config.json": "e8cc253fee3fdd6c27d9d446d6200ca8a12d1c64741ea7ea976df2f9bc6c5969",
            "model.safetensors": "13aec07f51e257e8deb4208855bb9f0754ac368aeb674e6a12f82427d3f15dde",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "22fd4a60565f25fee8ebd2866aab07456dd46cbbeb45eaf506529ed8451ee58f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Modality",
        "5384b8997e3cbb79ca3a670e869577f4e4f4997e",
        "7416e2da325c9b8f01f7fe7d7a4c89060a6f101504206d9f58da5150bfc7fa9b",
        307_532_547,
        {
            "config.json": "c82b9723eefe61051db8f65b50a4ab69efd3ee2bdd5ea88f80d2a42460594742",
            "model.safetensors": "27d50c59a94e118643a2a53e47aa08ef9f7c3e0f5b1fb5b1f43a5f46ad134d2b",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Hazard",
        "5dd25f2cc3c98f338e6a79b667662d60f936a28d",
        "f8c07d8b82765062d79bb5ee67651a1cb27d8d8012d7adb9ba7b89b190785102",
        307_539_468,
        {
            "config.json": "d43f6168eccfcc96cde59b6a7578653f07060c6525cd4a9c49b049d94faa824d",
            "model.safetensors": "01c044a3fff1e2d6b3a12a90e2847e714184884931b610a2108407f5d84213f2",
            "operating_point.json": "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "22fd4a60565f25fee8ebd2866aab07456dd46cbbeb45eaf506529ed8451ee58f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "PII",
        "6d3300c4bd7975f30a664503f6c725cf1fbbad48",
        "f4fd4a1d9993954a15e8ec5df4de5f70ae717f39f89156ffa2fd989e90697dbb",
        307_557_155,
        {
            "config.json": "79eca4e6ab4563c983380ded17bbf66eae16c017b5a4df050de61bb8fb9c9855",
            "model.safetensors": "79053320ba936f78e0b870f70e21b334c749ca19bb91e6d5b95cda3400f358aa",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Halu",
        "ca87531211e414ac21c641b2faa8b8e21619de8f",
        "8bc0a7b87a5b49aaeb28506f122c4933802452f7422e6aec2e3f8dce00a0e1e5",
        307_531_778,
        {
            "config.json": "69af1ca2e384fab95b30f2660f624e8febfa126a1f7aabef8669679727c3cfd2",
            "model.safetensors": "0f51bf4a6e1462e88733c36e1e59fa96c1188c8c470de7829b51c95298247be6",
            "operating_point.json": "eabd44bed728779ed89afdefbc7f42864daa87aca8ace7dc24fd207b40094a5c",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "0be5487f39c5ddc334beb659aa180e4d8a347293bc21a5d85df7ec1674cc4c8c",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Embedding",
        "1e57cebf5a7b7fec6e6973f05bbca97c5cca4436",
        "163812d40a9eddd55fe9430111d2f36a81c0a3a6fa42c8e80393c551187a6ce1",
        306_939_648,
        {
            "1_Pooling/config.json": "6401a8eeaeca1f1fe2b1b46be987dcf2ee6085ec73f7d2263318ffc191dd91d3",
            "config.json": "8ac04bd0a7c651189ffd749bae3482551aac6c20e7fc3e09a4a795df228882ed",
            "model.safetensors": "42a1761bfd1831661d073729f8d04657bb11f543b233b951d2f4e5db581ece64",
            "modules.json": "21ddb3037a55ebf4549eb5c1dc44c3f28f10b56ec7f1335c92fd90e5b30d88ac",
            "onnx/model.onnx": "f120cc4064990b5ed902759e59c6bb9a431ec531edadfe629fa938ac7b96b131",
            "onnx/model_layer_11.onnx": "5c8a7b1ca12af43f8bcb94a9a5e85264159b183d57961eb6c3d4cd3844160b2e",
            "onnx/model_layer_3.onnx": "255ecfa892d6595cb6c5d6dac17d11c33810eda5051e938e5404e91bd4cf322a",
            "onnx/model_layer_6.onnx": "88f9b1857c5643347888f036eaef283aa38d2ea87a43f387fa4c7e243517a087",
            "onnx/weights.data": "5eab25f517b84ab562bbcf10978275e7347c2d483107bcb7ddccb29618ad243b",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    _vela(
        "Reranker",
        "a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf",
        "ee533204062166c6c0494a4fec56f935e4ef3b071bc245f5d8fbb2517e38675d",
        306_939_648,
        {
            "classification_heads.safetensors": "138d8192a406367da628fba41fe6227f237b982038e09686148dd4d56b4cef5d",
            "config.json": "5ba550d725efc4bfff1737546c4e3efc9eae67a49e2ce213bbaf78b00e41b1c8",
            "matryoshka_config.json": "69bca4f7672b045bda1ceeb9f989e559164971797b0e0d6a5f48ca7f8631454c",
            "model.safetensors": "582a6aaf7eeb71a03246a9b42be2a221f82e9b5d0a173570d3eeab8118f1662d",
            "onnx/model.onnx": "92249512660cf7fa4dffb8993c5c840c6f4453cdbe8a850ad08f8d84a2701c22",
            "onnx/model_layer_11_dim_128.onnx": "fea94206aea36c4cfc73a5e3131cceaa9a6cc8d3512affca637507b7d2cc821b",
            "onnx/model_layer_11_dim_256.onnx": "671230c541e40d1e80c0fdf86a1f0cf4e780c516d9bbd18d0be113345daa8c3d",
            "onnx/model_layer_11_dim_512.onnx": "0de6094ccda35e41f0cd024c21509de41890319652c4b62eea2adb80dedd3e3a",
            "onnx/model_layer_11_dim_64.onnx": "d9cf6bc3e3750ca1d7e831688ae7e62c3adfcd15027abededd64aae674d42a45",
            "onnx/model_layer_11_dim_768.onnx": "62b5beb74a9d665e0322e19218a7248ce1e5416ed420cced1428d765324367c2",
            "onnx/model_layer_22_dim_128.onnx": "4aac3f2adb4d2f33a7b701b86a4559bcef19192ff38545ed29dbb8093be592a6",
            "onnx/model_layer_22_dim_256.onnx": "cd75f62254245bf54a75ee730b901ee288a85fa5898eb0f4b336ab766a8a8ad3",
            "onnx/model_layer_22_dim_512.onnx": "41e91dff631d463b40225129d9a970daf5578d50b1b66650d4a5bef942555d4e",
            "onnx/model_layer_22_dim_64.onnx": "2953319eec426ca655f33ee6bfdd5a7abc52664501deb21cca0ffd44fdbb7188",
            "onnx/model_layer_3_dim_128.onnx": "6c95df1a0541b2dd2e2123462d51edbc0210a305ba0fe241e52a3221fc5c23cf",
            "onnx/model_layer_3_dim_256.onnx": "915b5ff7ece9a572e1633e9bc08916b8893096e41ac086a008c7d3260143b2fc",
            "onnx/model_layer_3_dim_512.onnx": "dc821fd4c3842a858e0fb315708b228d53a0261bc67f1482695e80aade9e50f0",
            "onnx/model_layer_3_dim_64.onnx": "295227573b762595185f99d049d03a0dd9cfae069d546b7dcf2ff7249e546726",
            "onnx/model_layer_3_dim_768.onnx": "00abc7b16c2d516d0a9cbe69bd25a52799147d0c1662e455a6db950c354ce977",
            "onnx/model_layer_6_dim_128.onnx": "3fcf58280ed30f9b264ca90ea5bbcd7696136a1c2fdb4133378341d05a555700",
            "onnx/model_layer_6_dim_256.onnx": "9cf85507788958fd8f85dc36975a204448ba9127a49a9cbf635407208371798b",
            "onnx/model_layer_6_dim_512.onnx": "c764046275644c54fdea0458c068c138443e1de918e173fb99b438b7c4a58415",
            "onnx/model_layer_6_dim_64.onnx": "29f45c33bd1b20043c7e8fc8fa69ab65a46708220b2b38284d804a8ecbb36f6a",
            "onnx/model_layer_6_dim_768.onnx": "f7e44de027ff783aedc52c39fc1ac027f71a061f3e874768382558d911d130ad",
            "onnx/weights.data": "84ef3c480a28e6fd4f8da5eb22bdd6646eef76d043c01575af6f4db7cde3179b",
            "special_tokens_map.json": "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903",
            "tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
            "tokenizer_config.json": "74a259bb1a3811a7e3028adcd07a65765d866d5e66e0e883f0994ccfa67e8455",
        },
    ),
    BuiltinModel(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        family="task_heads",
        model_sha256="dd27783a75562fa0a720cbdce47cca00ed95742be2018a6c93a67ffd2c782956",
        manifest_sha256="",
        loaded_parameters=595_776_512,
        backbone="qwen3",
        min_device_memory_gib=4,
        files={
            "1_Pooling/config.json": "37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd",
            "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
            "config_sentence_transformers.json": "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
            "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
            "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
            "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
            "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
        },
    ),
)
MODELS = with_recorded(MODELS, "golden_answers_vela1.json", "answers", "golden_answers")
