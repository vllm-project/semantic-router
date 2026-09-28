use super::*;

fn qwen_fixture() -> TempDir {
    let dir = fixture(&["safe", "unsafe"], 0);
    std::fs::write(
        dir.path().join("config.json"),
        json!({
            "model_type":"qwen3", "vocab_size":16,"hidden_size":4,"intermediate_size":8,
            "num_hidden_layers":1,"num_attention_heads":1,"head_dim":4,"attention_bias":false,
            "num_key_value_heads":1,"max_position_embeddings":2048,"sliding_window":null,
            "max_window_layers":1,"tie_word_embeddings":true,"rope_theta":10000.0,
            "rms_norm_eps":0.00001,"use_sliding_window":false,"hidden_act":"silu"
        })
        .to_string(),
    )
    .unwrap();
    let mut tensors = HashMap::new();
    for (name, shape) in [
        ("model.embed_tokens.weight", vec![16, 4]),
        ("model.norm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.q_norm.weight", vec![4]),
        ("model.layers.0.self_attn.k_norm.weight", vec![4]),
        ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
    ] {
        tensors.insert(
            name.to_owned(),
            Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
    dir
}

#[test]
fn owned_generative_preserves_conditional_scores_and_checks_full_prompt_budget() {
    let dir = qwen_fixture();
    let mut opts = options(&dir);
    opts.model_type = "qwen3".into();
    opts.overflow = "reject".into();
    let a = load(opts.clone(), "generative").unwrap();
    let b = load(opts.clone(), "generative").unwrap();
    let labels = vec!["hello".to_owned(), "world".to_owned()];
    let output = value(
        a.generative("hello world", None, labels.clone(), false)
            .unwrap(),
    );
    assert_eq!(output["score_semantics"], "first_token_label_softmax");
    assert_eq!(output["adapter_weights_applied"], false);
    assert!(output["input"]["input_tokens"].as_u64().unwrap() > 2);
    drop(a);
    assert_eq!(
        output,
        value(
            b.generative("hello world", None, labels.clone(), false)
                .unwrap()
        )
    );
    assert!(b
        .generative("hello", Some("foreign"), vec![], false)
        .is_err());
    opts.max_input_tokens = 4;
    let limited = load(opts, "generative").unwrap();
    assert!(limited.generative("hello", None, labels, false).is_err());
}

#[test]
fn owned_guard_counts_prefix_and_suffix_and_survives_peer_close() {
    let dir = qwen_fixture();
    let mut opts = options(&dir);
    opts.model_type = "qwen3".into();
    opts.overflow = "reject".into();
    opts.generation_max_tokens = 2;
    let a = load(opts.clone(), "guard").unwrap();
    let b = load(opts, "guard").unwrap();
    let output = value(a.guard("hello world", "input").unwrap());
    assert!(output["input"]["input_tokens"].as_u64().unwrap() > 2);
    drop(a);
    assert_eq!(output, value(b.guard("hello world", "input").unwrap()));
    assert!(b.guard("hello", "invalid").is_err());
}
