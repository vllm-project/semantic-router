use super::mmbert_embedding::MmBertEmbeddingModel;
use super::runtime_identity::ArtifactSnapshot;
use std::path::Path;
use tokenizers::{models::wordpiece::WordPiece, pre_tokenizers::whitespace::Whitespace, Tokenizer};

// A real, tiny ONNX graph. Avoid downloading weights or generating a second
// inference implementation: ORT loads this Constant initializer graph itself.
fn varint(mut value: u64) -> Vec<u8> {
    let mut output = vec![];
    while value >= 128 {
        output.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    output.push(value as u8);
    output
}
fn integer(number: u64, value: u64) -> Vec<u8> {
    [varint(number << 3), varint(value)].concat()
}
fn field(number: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(number << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn value_info(name: &str, dtype: u64, dimensions: &[u64]) -> Vec<u8> {
    let shape = dimensions
        .iter()
        .flat_map(|&dimension| field(1, &integer(1, dimension)))
        .collect::<Vec<_>>();
    let tensor_type = [integer(1, dtype), field(2, &shape)].concat();
    [field(1, name.as_bytes()), field(2, &field(1, &tensor_type))].concat()
}
fn tiny_graph() -> Vec<u8> {
    let tensor = [
        integer(1, 1),
        integer(1, 2),
        integer(1, 64),
        integer(2, 1),
        field(8, b"hidden"),
        field(9, &vec![0; 2 * 64 * 4]),
    ]
    .concat();
    let graph = [
        field(2, b"identity-test"),
        field(5, &tensor),
        field(11, &value_info("input_ids", 7, &[1, 2])),
        field(11, &value_info("attention_mask", 7, &[1, 2])),
        field(12, &value_info("hidden", 1, &[1, 2, 64])),
    ]
    .concat();
    [integer(1, 8), field(7, &graph), field(8, &integer(2, 13))].concat()
}
fn model_files(path: &Path) {
    std::fs::write(path.join("config.json"),serde_json::json!({"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"max_position_embeddings":128}).to_string()).unwrap();
    let vocab = [("[UNK]".to_string(), 0), ("hello".to_string(), 1)];
    let mut tokenizer = Tokenizer::new(
        WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer.save(path.join("tokenizer.json"), false).unwrap();
    let mut preferred = tiny_graph();
    preferred[1] = 127; // Valid protobuf, unsupported ONNX IR: fail inside ORT.
    std::fs::write(path.join("model.onnx"), preferred).unwrap();
    std::fs::write(path.join("model_optimized.onnx"), tiny_graph()).unwrap();
}
#[test]
fn descriptor_uses_successful_graph_and_actual_provider_after_fallback() {
    let dir = tempfile::tempdir().unwrap();
    model_files(dir.path());
    let mut model = MmBertEmbeddingModel::load(dir.path(), true).unwrap();
    let descriptor = model.runtime_descriptor(0, 0).unwrap();
    let selected =
        ArtifactSnapshot::capture(&dir.path().join("model_optimized.onnx"), "graph").unwrap();
    assert_eq!(descriptor.artifacts[0].sha256, selected.digest.sha256);
    assert_eq!(descriptor.runtime, "onnx-mmbert-cpu-v1");
    assert_eq!(descriptor.layer, 2);
    assert_eq!(descriptor.dimension, 64);
    assert_eq!(
        serde_json::to_value(&descriptor).unwrap(),
        serde_json::to_value(model.runtime_descriptor(2, 64).unwrap()).unwrap()
    );
    assert!(model.runtime_descriptor(1, 64).is_err());
    assert!(model.runtime_descriptor(2, 65).is_err());
    assert_eq!(
        model.encode(&["hello hello"], None, None).unwrap().shape(),
        &[1, 64]
    );
    std::fs::write(
        dir.path().join("model_optimized.onnx"),
        b"later path replacement",
    )
    .unwrap();
    assert_eq!(
        model.runtime_descriptor(0, 0).unwrap().artifacts[0].sha256,
        selected.digest.sha256
    );

    #[cfg(not(any(feature = "cuda", feature = "rocm", feature = "migraphx")))]
    {
        std::fs::write(dir.path().join("model_optimized.onnx"), tiny_graph()).unwrap();
        let requested_gpu = MmBertEmbeddingModel::load(dir.path(), false).unwrap();
        assert_eq!(
            requested_gpu.runtime_descriptor(0, 0).unwrap().runtime,
            "onnx-mmbert-cpu-v1"
        );
    }
}
