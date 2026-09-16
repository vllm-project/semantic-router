use super::mmbert_embedding::{MmBertEmbeddingConfig, MmBertEmbeddingModel};
use crate::model_architectures::model_factory::ModelFactory;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;
use tokenizers::{models::wordpiece::WordPiece, Tokenizer};

fn tiny_model(path: &Path) {
    let config = serde_json::json!({"vocab_size":16,"hidden_size":64,"num_hidden_layers":2,
        "num_attention_heads":4,"intermediate_size":128,"max_position_embeddings":128,
        "local_attention":8,"global_attn_every_n_layers":3,"global_rope_theta":160000.0,
        "local_rope_theta":160000.0});
    std::fs::write(path.join("config.json"), config.to_string()).unwrap();
    let config = MmBertEmbeddingConfig::from_pretrained(path).unwrap();
    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    MmBertEmbeddingModel::load_with_vb(path.to_str().unwrap(), &config, vb, &Device::Cpu).unwrap();
    variables.save(path.join("model.safetensors")).unwrap();
    let vocab = [("[UNK]".to_string(), 0), ("hello".to_string(), 1)];
    let tokenizer = Tokenizer::new(
        WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.save(path.join("tokenizer.json"), false).unwrap();
}

fn load(path: &Path) -> ModelFactory {
    let mut factory = ModelFactory::new(Device::Cpu);
    factory
        .register_mmbert_embedding_model(path.to_str().unwrap())
        .unwrap();
    factory
}

#[test]
fn actual_loaded_identity_tracks_content_and_effective_exit_not_path_or_docs() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    tiny_model(first.path());
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        std::fs::copy(first.path().join(name), second.path().join(name)).unwrap();
    }
    std::fs::write(
        second.path().join("README.md"),
        "different revision and docs",
    )
    .unwrap();
    let factory = load(first.path());
    let original = factory.mmbert_runtime_descriptor(0, 0).unwrap();
    let copied = load(second.path())
        .mmbert_runtime_descriptor(2, 64)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&original).unwrap(),
        serde_json::to_value(copied).unwrap()
    );
    assert_eq!(original.layer, 2);
    assert_eq!(original.dimension, 64);
    assert_eq!(factory.mmbert_runtime_descriptor(1, 32).unwrap().layer, 1);
    assert_eq!(
        factory.mmbert_runtime_descriptor(1, 32).unwrap().dimension,
        32
    );
    assert!(factory.mmbert_runtime_descriptor(3, 64).is_err());
    assert!(factory.mmbert_runtime_descriptor(2, 65).is_err());

    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(second.path().join("config.json")).unwrap()).unwrap();
    config["_name_or_path"] = serde_json::json!("an unrelated local path");
    config["_commit_hash"] = serde_json::json!("documentation-only-revision");
    std::fs::write(second.path().join("config.json"), config.to_string()).unwrap();
    assert_eq!(
        original.effective_config_sha256,
        load(second.path())
            .mmbert_runtime_descriptor(0, 0)
            .unwrap()
            .effective_config_sha256
    );
    config["global_rope_theta"] = serde_json::json!(20000.0);
    std::fs::write(second.path().join("config.json"), config.to_string()).unwrap();
    assert_ne!(
        original.effective_config_sha256,
        load(second.path())
            .mmbert_runtime_descriptor(0, 0)
            .unwrap()
            .effective_config_sha256
    );

    let tokenizer_path = second.path().join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
    tokenizer.add_tokens(&[tokenizers::AddedToken::from("new-token", false)]);
    tokenizer.save(tokenizer_path, false).unwrap();
    assert_ne!(
        original.tokenizer_sha256,
        load(second.path())
            .mmbert_runtime_descriptor(0, 0)
            .unwrap()
            .tokenizer_sha256
    );

    let weights_path = first.path().join("model.safetensors");
    let mut weights = std::fs::read(&weights_path).unwrap();
    let offset = weights.len() - 4;
    weights[offset..].copy_from_slice(&0.123_f32.to_le_bytes());
    std::fs::write(weights_path, weights).unwrap();
    assert_ne!(
        original.artifacts[0].sha256,
        load(first.path())
            .mmbert_runtime_descriptor(0, 0)
            .unwrap()
            .artifacts[0]
            .sha256
    );
    // An already initialized factory keeps its captured identity, even if the
    // requested directory is replaced later. It does not re-read that path.
    assert_eq!(
        original.artifacts[0].sha256,
        factory.mmbert_runtime_descriptor(0, 0).unwrap().artifacts[0].sha256
    );
}
