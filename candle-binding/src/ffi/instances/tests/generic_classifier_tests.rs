use super::*;
use crate::ffi::generic_classifier::GenericClassifier;
use crate::ffi::{
    classify_text, classify_text_with_probabilities, free_probabilities, init_generic_classifier,
};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::ffi::CString;

fn bert_fixture() -> TempDir {
    let dir = fixture(&["weather", "travel", "other"], 2);
    let config = json!({
        "model_type": "bert", "vocab_size": 16, "hidden_size": 4,
        "num_hidden_layers": 1, "num_attention_heads": 1, "intermediate_size": 8,
        "hidden_act": "gelu", "hidden_dropout_prob": 0.0,
        "max_position_embeddings": 512, "type_vocab_size": 2,
        "initializer_range": 0.02, "layer_norm_eps": 1e-12, "pad_token_id": 0
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let config: BertConfig = serde_json::from_value(config).unwrap();
    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
    BertModel::load(vb.pp("bert"), &config).unwrap();
    candle_nn::linear(
        config.hidden_size,
        config.hidden_size,
        vb.pp("bert.pooler.dense"),
    )
    .unwrap();
    candle_nn::linear(config.hidden_size, 3, vb.pp("classifier")).unwrap();
    variables
        .save(dir.path().join("model.safetensors"))
        .unwrap();
    dir
}

#[test]
fn generic_classifier_selects_modernbert_family_from_config() {
    for model_type in ["modernbert", "mmbert", "mmbert32k", "mmbert-32k"] {
        let dir = fixture(&["weather", "travel", "other"], 2);
        let config_path = dir.path().join("config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["model_type"] = json!(model_type);
        std::fs::write(config_path, config.to_string()).unwrap();
        let path = dir.path().to_str().unwrap();
        let classifier = GenericClassifier::new(path, 3, true).unwrap();
        let GenericClassifier::ModernBert(model) = &classifier else {
            panic!("{model_type} selected the BERT loader");
        };
        assert!(model.device().is_cpu());
        let expected = TraditionalModernBertClassifier::load_from_directory(path, true)
            .unwrap()
            .classify_text_with_probabilities("hello world")
            .unwrap();
        assert_eq!(
            classifier
                .classify_text_with_probabilities("hello world")
                .unwrap(),
            expected
        );
        assert_eq!(expected.0, 2);
        assert_eq!(expected.2.len(), 3);
        assert!((expected.2.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(GenericClassifier::new(path, 2, true)
            .unwrap_err()
            .to_string()
            .contains("number of classes"));
    }
}

#[test]
fn generic_classifier_preserves_classic_bert_and_explicit_class_count() {
    let dir = bert_fixture();
    let path = dir.path().to_str().unwrap();
    let expected = TraditionalBertClassifier::new(path, 3, true)
        .unwrap()
        .classify_text_with_probabilities("hello world")
        .unwrap();
    let config_path = dir.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    for model_type in [Some(json!("bert")), None] {
        if let Some(model_type) = model_type {
            config["model_type"] = model_type;
        } else {
            config.as_object_mut().unwrap().remove("model_type");
        }
        std::fs::write(&config_path, config.to_string()).unwrap();
        let classifier = GenericClassifier::new(path, 3, true).unwrap();
        let GenericClassifier::Bert(model) = &classifier else {
            panic!("classic BERT selected the ModernBERT loader");
        };
        assert!(model.device().is_cpu());
        assert_eq!(
            classifier
                .classify_text_with_probabilities("hello world")
                .unwrap(),
            expected
        );
        assert_eq!(
            classifier.classify_text("hello world").unwrap(),
            (expected.0, expected.1)
        );
        assert_eq!(expected.2.len(), 3);
        assert!(GenericClassifier::new(path, 2, true).is_err());
    }
}

#[test]
fn generic_classifier_rejects_unsupported_or_invalid_model_type() {
    let dir = fixture(&["weather", "travel"], 0);
    let path = dir.path().to_str().unwrap();
    let config_path = dir.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    for model_type in [json!("roberta"), json!(42)] {
        config["model_type"] = model_type;
        std::fs::write(&config_path, config.to_string()).unwrap();
        assert!(GenericClassifier::new(path, 2, true)
            .unwrap_err()
            .to_string()
            .contains("model_type"));
    }
}

#[test]
fn generic_classifier_ffi_returns_full_distribution_after_failed_load() {
    let dir = fixture(&["weather", "travel", "other"], 2);
    let path = CString::new(dir.path().to_str().unwrap()).unwrap();
    assert!(!init_generic_classifier(path.as_ptr(), 2, true));
    assert!(init_generic_classifier(path.as_ptr(), 3, true));
    let text = CString::new("hello world").unwrap();
    let top = classify_text(text.as_ptr());
    let result = classify_text_with_probabilities(text.as_ptr());
    assert_eq!(result.num_classes, 3);
    assert!(!result.probabilities.is_null());
    let probabilities = unsafe { std::slice::from_raw_parts(result.probabilities, 3).to_vec() };
    // SAFETY: this live result owns the matching allocation and is released once.
    unsafe { free_probabilities(result.probabilities, result.num_classes) };
    assert!(result.label.is_null());
    assert_eq!(result.predicted_class, 2);
    assert_eq!(top.predicted_class, result.predicted_class);
    assert_eq!(top.confidence, result.confidence);
    assert_eq!(result.confidence, probabilities[2]);
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}
