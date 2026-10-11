"""Decision 3.0: the released d3 packages, pinned by revision and identity.

Each entry downloads only the files the decision3 family loads (the weights,
readout, configs, tokenizer, chat template and image processor settings, and
the manifest); the packages' bundled Python is never fetched.
"""

from __future__ import annotations

from .common import ORG, BuiltinModel, with_recorded

DECISION3_MODELS: tuple[BuiltinModel, ...] = (
    BuiltinModel(
        repo_id=f"{ORG}/d3",
        revision="dc6c41cb98429ea8fc30e5bfb04b75757222575c",
        family="decision3",
        model_sha256="a44afeb0dbd9849dde06d4e477f69bcf8709620f8d0659ec317d6b9998be2681",
        manifest_sha256="474a7a4d5dfcba4dfc5330db5fcb465cf29aee1db4a7c99dbe44c32a09c590ac",
        loaded_parameters=26_086_635_760,
        backbone="qwen3_5_text",
        min_device_memory_gib=72,
        files={
            "MODEL_MANIFEST.json": "474a7a4d5dfcba4dfc5330db5fcb465cf29aee1db4a7c99dbe44c32a09c590ac",
            "config.json": "7e16284fafd2d54b73c10073c0391bfd6804886ceaf2cbe37fdb62d4c6813dea",
            "decision_config.json": "6b80ca11bd6ba481df4b3c187db8a5d1983786563ca05e2edccb0bdac0d2fc4f",
            "readout.safetensors": "59b64e75ef8b7bf1da667224cfcac3505901aee4d3ca87efd607f4495b04c073",
            "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
            "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
            "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "model-00001-of-00011.safetensors": "fc8e8d80540484dcfe9fc4915e7bed0da5b096d93018feb15dd9891462cfcd38",
            "model-00002-of-00011.safetensors": "1b33617dc6ed04b9e9a927cf181e6ad3dd33db90a3f6b6ca244b54b11be758a3",
            "model-00003-of-00011.safetensors": "2141d2f24924f2206b1fb7080ad0a87105051ca0b89bccf43444491f39045234",
            "model-00004-of-00011.safetensors": "d6061e5c3c450e23cf994ab4180344ec5650bc8f70909aab70daa4037dd4f3d2",
            "model-00005-of-00011.safetensors": "1fb7c2b192c8bf6276f75e241b9f66b60887458fe9d03053fa2d0b044cc9c5cd",
            "model-00006-of-00011.safetensors": "155511100c14cfcca7b6c8c34880671e16897908330e7f079958bd69e25eb826",
            "model-00007-of-00011.safetensors": "87177d5979c45ead4417510b32ffb2b9ba3af7803d22d683c138b06695850acf",
            "model-00008-of-00011.safetensors": "612d01d8a0fc5c9c8244d8e8b674e96d0803a4e6c590a1897963754709af199a",
            "model-00009-of-00011.safetensors": "3cc1b3ca87ce53983add341a32938a3a871593b086b20d1929f4782b692b962a",
            "model-00010-of-00011.safetensors": "4488b64fd5b81b9a1df80a3685e88e413f120499eadc7772b0927cb2e6d21cdc",
            "model-00011-of-00011.safetensors": "ebf7c95fabb345495fad5fbe6c6c75c29562d4b5891e8aea116cce46df8a68ef",
            "model.safetensors.index.json": "dc39d771cb240f775399e1845b11d081d8e007f14eecae3b780bc6272b128a46",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/d3-flash",
        revision="581c9953d43579645f6249e91430518ee7d838a1",
        family="decision3",
        model_sha256="74b71d370f3d9e5731c603dafcb8af4e9134fd2c769c89c5169b031a0418799b",
        manifest_sha256="f7eecfae59b0b35763a75079f11e4446d278be7718f482932ff5290145f8bfe3",
        loaded_parameters=8_393_739_504,
        backbone="qwen3_5_text",
        min_device_memory_gib=32,
        files={
            "MODEL_MANIFEST.json": "f7eecfae59b0b35763a75079f11e4446d278be7718f482932ff5290145f8bfe3",
            "config.json": "9c3e6eea217964096c7b032285bab6757720a847fbb9d0a04e551d2b1606c0eb",
            "decision_config.json": "0269c72ac055d8149a7a778c477780af361f4c1cde3fd0e5ba9af791453ece18",
            "readout.safetensors": "8bbc9a9df4e1b7df85a80a1091977092b938d8d2d2869dfaeb8fd36063cf322f",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "model-00001-of-00004.safetensors": "2c322016b91b1232c0d7628560164560d06d477a3edc3362714635f0e2f5949c",
            "model-00002-of-00004.safetensors": "f769650fb747d85132fb321335af7c71fcf3ba95fd1b1e806447e891cb4b0546",
            "model-00003-of-00004.safetensors": "c85a0f2f8df221b7864c74e778d802dc22fe1237fe2fa6ea8b81d7b612800a33",
            "model-00004-of-00004.safetensors": "d686ef3f6cdeeaad5eedde9d516c29ed1c90a1a322dd002765732869c11fa269",
            "model.safetensors.index.json": "9ddd9f9ce431ff771d9b37b354ba11aa1b74c18d87c5a48c964da794bd1029ee",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/d3-mini",
        revision="61dbd3a320591004ade9afa3cc95cddae78a2488",
        family="decision3",
        model_sha256="9ac186bb30b30c129d7aae03d73932600424963bd4644c2041a9d7120f029af5",
        manifest_sha256="f1f94469cf73c58019fb3f9c4e7172bba556794a1f8160030285cdca81a091d5",
        loaded_parameters=4_539_918_336,
        backbone="qwen3_5_text",
        min_device_memory_gib=16,
        files={
            "MODEL_MANIFEST.json": "f1f94469cf73c58019fb3f9c4e7172bba556794a1f8160030285cdca81a091d5",
            "config.json": "8f5f404a71ecc53c4ff26fc42cb4c778ffacfb4c9623f5e2d51245166d6b5057",
            "decision_config.json": "588bc0ac7644479b9adde0498d80b6fb7c576388992564ac67599c05d29add07",
            "readout.safetensors": "d570665c1da72cb9fca7ad7b23a25720eb691362694dd69d4c80c17d7656ed55",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "model-00001-of-00002.safetensors": "60621cb8c3f90e634b98f0ffaf609159ee2c450630386f09dd5f7fd7938d7598",
            "model-00002-of-00002.safetensors": "6189569fbe9245030d7ffb0b60a52268b7386e7a85a57aee1274a5178c1cc53d",
            "model.safetensors.index.json": "613cc71f6441a964b0a87470536f8fb0c639bdd86b0362027c1c2b1a6e0a4a8c",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/d3-nano",
        revision="6601b4d1398bdbdfa8813dff456a96b92458ca95",
        family="decision3",
        model_sha256="1c54342d7bcab9516a9d5d6ff13ef865ecac38e044658998004429aeb5ab57dc",
        manifest_sha256="24b9240c62cce9f5ddc1e2cc9ecc309e10982dbd4c49487291a902bef998869b",
        loaded_parameters=2_213_763_904,
        backbone="qwen3_5_text",
        min_device_memory_gib=8,
        files={
            "MODEL_MANIFEST.json": "24b9240c62cce9f5ddc1e2cc9ecc309e10982dbd4c49487291a902bef998869b",
            "config.json": "9caa6ab8956c859c3852328dda6322893a18b30826d3916db91d938d6c74ac4f",
            "decision_config.json": "9175a0f42d5886def9c8429fdf8b0be58a3c97d106b94520e2ac38fe326302bf",
            "readout.safetensors": "4bf6300d52b87d55bc458f8728cde10e8d6ee916633f15ef23ec0a855aeec5f3",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "model.safetensors": "b228dbb2a3af639c7be53256fcf39c724acd50103522ee0832017c4b702d3467",
        },
    ),
    BuiltinModel(
        repo_id=f"{ORG}/d3-lite",
        revision="b731454b7b17e455cbda511574a274a731ea8c2e",
        family="decision3",
        model_sha256="9499da7624f8f89c7b77a66dbc4b31745de63ca2034fe5852b5db8cfb2da4e20",
        manifest_sha256="360003d11bd31b0d65858b1cfb8e02231116fcbe7182651fc8dbc7bcb58fb5bf",
        loaded_parameters=853_247_040,
        backbone="qwen3_5_text",
        min_device_memory_gib=4,
        files={
            "MODEL_MANIFEST.json": "360003d11bd31b0d65858b1cfb8e02231116fcbe7182651fc8dbc7bcb58fb5bf",
            "config.json": "a9071195384e1fc2c68f9fe0fae007a22182ab77eb14fe8228a1a24b37d3cee9",
            "decision_config.json": "30b6095b945d5a93697868afdf9be9123ea892085fbaff3078112c48da20dc6b",
            "readout.safetensors": "0435e68c4146f0e6c3aa7108e37ea7d327f184e3dc7c3d9f98cefe580dc9ceb6",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "model.safetensors": "53cc902344417a33e776a1d4b6ce7596034aba51854fd259a4a7b179ee1d03fe",
        },
    ),
)

DECISION3_MODELS = with_recorded(
    DECISION3_MODELS, "golden_answers_decision3.json", "answers", "golden_answers"
)
DECISION3_MODELS = with_recorded(
    DECISION3_MODELS, "kernel_choices.json", "choices", "kernel_choices"
)

MODELS = DECISION3_MODELS
