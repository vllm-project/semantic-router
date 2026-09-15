//! No model downloads: each case loads a tiny local mmBERT checkpoint through
//! the real FFI entrypoints. A fresh process per case isolates the OnceLocks.
//! An empty ModelFactory represents ownership by an unrelated embedding model;
//! its weights are irrelevant to the mmBERT registration decision. The separate
//! gate case exercises all four real factory-owning FFI entrypoints, including
//! multimodal, without downloading a production multimodal checkpoint.

use super::*;
use crate::model_architectures::embedding::mmbert_embedding::MmBertEmbeddingConfig;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::ffi::CString;
use std::process::{Command, Stdio};
use std::sync::{mpsc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const CHILD_CASE: &str = "SEMANTIC_ROUTER_EMBEDDING_INIT_TEST_CASE";

fn fixture(hidden_size: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config_json = serde_json::json!({
        "vocab_size": 4, "hidden_size": hidden_size, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": hidden_size * 2,
        "max_position_embeddings": 16, "layer_norm_eps": 1e-5,
        "pad_token_id": 0, "global_attn_every_n_layers": 3,
        "global_rope_theta": 10000.0, "local_attention": 8,
        "local_rope_theta": 10000.0
    });
    std::fs::write(dir.path().join("config.json"), config_json.to_string()).unwrap();
    let config = MmBertEmbeddingConfig::from_pretrained(dir.path()).unwrap();
    let vars = VarMap::new();
    let model = MmBertEmbeddingModel::load_with_vb(
        dir.path().to_str().unwrap(),
        &config,
        VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        &Device::Cpu,
    )
    .unwrap();
    drop(model);
    vars.save(dir.path().join("model.safetensors")).unwrap();
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("hello".into(), 1),
                ("world".into(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    MmTokenizer::new(wordlevel)
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    dir
}

fn path(dir: &tempfile::TempDir) -> CString {
    CString::new(dir.path().to_str().unwrap()).unwrap()
}

fn combined(path: &CString, use_cpu: bool) -> bool {
    init_embedding_models_with_mmbert(std::ptr::null(), std::ptr::null(), path.as_ptr(), use_cpu)
}

fn install_other_factory() {
    let _guard = lock_embedding_init().unwrap();
    assert!(GLOBAL_MODEL_FACTORY
        .set(ModelFactory::new(Device::Cpu))
        .is_ok());
}

fn assert_embeddings(hidden_size: usize) {
    let (model, _) = get_mmbert_refs().expect("mmBERT must remain reachable");
    assert_eq!(model.config().hidden_size, hidden_size);
    let factory = GLOBAL_MODEL_FACTORY.get().unwrap();
    let one = generate_mmbert_embedding(factory, "hello", None, None).unwrap();
    assert_eq!(one.len(), hidden_size);
    assert!(one.iter().all(|v| v.is_finite()));
    let batch = generate_mmbert_embeddings_batch(factory, &["hello", "world"], None, None).unwrap();
    assert_eq!(batch.len(), 2);
    assert!(batch
        .iter()
        .all(|row| row.len() == hidden_size && row.iter().all(|v| v.is_finite())));
}

fn assert_repeated_init_is_noop() {
    let (model, tokenizer) = get_mmbert_refs().unwrap();
    let model_ptr = model as *const _;
    let tokenizer_ptr = tokenizer as *const _;
    let second = fixture(16);
    let second_path = path(&second);
    assert!(combined(&second_path, false));
    assert!(init_mmbert_embedding_model(second_path.as_ptr(), false));
    // A missing path after successful initialization must not be opened at all.
    let missing = CString::new(second.path().join("missing").to_str().unwrap()).unwrap();
    assert!(combined(&missing, false));
    assert!(init_mmbert_embedding_model(missing.as_ptr(), false));
    let (same_model, same_tokenizer) = get_mmbert_refs().unwrap();
    assert_eq!(model_ptr, same_model as *const _);
    assert_eq!(tokenizer_ptr, same_tokenizer as *const _);
    assert_embeddings(8);
}

fn assert_all_entrypoints_share_gate() {
    let guard = lock_embedding_init().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    thread::scope(|scope| {
        let mut workers = Vec::new();
        for entry in 0..4 {
            let started = started_tx.clone();
            let done = done_tx.clone();
            workers.push(scope.spawn(move || {
                started.send(()).unwrap();
                let result = match entry {
                    0 => init_mmbert_embedding_model(std::ptr::null(), true),
                    1 => init_embedding_models_with_mmbert(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        true,
                    ),
                    2 => init_embedding_models(std::ptr::null(), std::ptr::null(), true),
                    _ => init_multimodal_embedding_model(std::ptr::null(), true),
                };
                done.send(result).unwrap();
            }));
        }
        for _ in 0..4 {
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert!(
            matches!(
                done_rx.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "an entrypoint bypassed the initialization gate"
        );
        drop(guard);
        for _ in 0..4 {
            assert!(!done_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        }
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(GLOBAL_MODEL_FACTORY.get().is_none());
}

fn assert_factory_first(case: &str) {
    let dir = fixture(8);
    install_other_factory();
    let model_path = path(&dir);
    assert!(if case == "factory_first" {
        combined(&model_path, true)
    } else {
        init_mmbert_embedding_model(model_path.as_ptr(), true)
    });
    assert!(GLOBAL_MODEL_FACTORY
        .get()
        .unwrap()
        .get_mmbert_model()
        .is_none());
    assert!(STANDALONE_MMBERT.get().is_some());
    assert_embeddings(8);
    assert_repeated_init_is_noop();
}

fn assert_mmbert_first(case: &str) {
    let dir = fixture(8);
    let model_path = path(&dir);
    assert!(if case == "mmbert_first" {
        combined(&model_path, true)
    } else {
        init_mmbert_embedding_model(model_path.as_ptr(), true)
    });
    // A later non-mmBERT initializer must not replace the winning factory.
    assert!(init_embedding_models(
        std::ptr::null(),
        std::ptr::null(),
        true
    ));
    assert!(GLOBAL_MODEL_FACTORY
        .get()
        .unwrap()
        .get_mmbert_model()
        .is_some());
    assert!(STANDALONE_MMBERT.get().is_none());
    assert_repeated_init_is_noop();
    assert!(
        STANDALONE_MMBERT.get().is_none(),
        "reinitialization must not load a duplicate"
    );
}

fn assert_concurrent_mmbert() {
    let dir = fixture(8);
    let barrier = Barrier::new(2);
    thread::scope(|scope| {
        let a = scope.spawn(|| {
            let model_path = path(&dir);
            barrier.wait();
            combined(&model_path, true)
        });
        let b = scope.spawn(|| {
            let model_path = path(&dir);
            barrier.wait();
            init_mmbert_embedding_model(model_path.as_ptr(), true)
        });
        assert!(a.join().unwrap());
        assert!(b.join().unwrap());
    });
    assert!(GLOBAL_MODEL_FACTORY
        .get()
        .unwrap()
        .get_mmbert_model()
        .is_some());
    assert!(
        STANDALONE_MMBERT.get().is_none(),
        "concurrent init must not load a duplicate"
    );
    assert_repeated_init_is_noop();
}

fn assert_concurrent_other_factory() {
    let dir = fixture(8);
    let guard = lock_embedding_init().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let model_path = path(&dir);
            started_tx.send(()).unwrap();
            done_tx.send(combined(&model_path, true)).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(
            GLOBAL_MODEL_FACTORY.get().is_none(),
            "the losing initializer must not load/publish early"
        );
        // Publish the competing factory while its initialization guard is
        // still held. The waiting mmBERT initializer must recheck ownership
        // after acquiring the guard and take the standalone path.
        assert!(GLOBAL_MODEL_FACTORY
            .set(ModelFactory::new(Device::Cpu))
            .is_ok());
        drop(guard);
        assert!(done_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        worker.join().unwrap();
    });
    assert!(STANDALONE_MMBERT.get().is_some());
    assert_embeddings(8);
}

fn assert_failed_load_retry(case: &str) {
    let dir = fixture(8);
    let model_path = path(&dir);
    if case == "failed_standalone_retry" {
        install_other_factory();
    }
    let tokenizer = dir.path().join("tokenizer.json");
    let bytes = std::fs::read(&tokenizer).unwrap();
    std::fs::remove_file(&tokenizer).unwrap();
    assert!(!combined(&model_path, true));
    assert!(
        get_mmbert_refs().is_none(),
        "failed init must not publish a partial model"
    );
    std::fs::write(tokenizer, bytes).unwrap();
    assert!(
        combined(&model_path, true),
        "failed initialization must be retryable"
    );
    assert_embeddings(8);
}

fn assert_poisoned_gate() {
    assert!(thread::spawn(|| {
        let _guard = EMBEDDING_INIT_LOCK.lock().unwrap();
        panic!("poison the initialization gate for this isolated case");
    })
    .join()
    .is_err());
    // Poison must be reported through the FFI bool, not an unwrap panic
    // that aborts the process across an extern C boundary.
    assert!(!init_mmbert_embedding_model(std::ptr::null(), true));
    assert!(!init_embedding_models_with_mmbert(
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        true
    ));
    assert!(!init_embedding_models(
        std::ptr::null(),
        std::ptr::null(),
        true
    ));
    assert!(!init_multimodal_embedding_model(std::ptr::null(), true));
}

fn run_case(case: &str) {
    assert!(GLOBAL_MODEL_FACTORY.get().is_none());
    assert!(STANDALONE_MMBERT.get().is_none());
    match case {
        "factory_first" | "factory_first_direct" => assert_factory_first(case),
        "mmbert_first" | "mmbert_first_direct" => assert_mmbert_first(case),
        "concurrent_mmbert" => assert_concurrent_mmbert(),
        "concurrent_other_factory" => assert_concurrent_other_factory(),
        "all_entrypoints_share_gate" => assert_all_entrypoints_share_gate(),
        "failed_load_retry" | "failed_standalone_retry" => assert_failed_load_retry(case),
        "poisoned_gate" => assert_poisoned_gate(),
        other => panic!("unknown initialization case: {other}"),
    }
}

#[test]
fn embedding_init_order_regressions() {
    if let Ok(case) = std::env::var(CHILD_CASE) {
        run_case(&case);
        return;
    }
    for case in [
        "factory_first",
        "factory_first_direct",
        "mmbert_first",
        "mmbert_first_direct",
        "concurrent_mmbert",
        "concurrent_other_factory",
        "all_entrypoints_share_gate",
        "failed_load_retry",
        "failed_standalone_retry",
        "poisoned_gate",
    ] {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!(
                    "{}::embedding_init_order_regressions",
                    module_path!().split_once("::").unwrap().1
                ),
                "--nocapture",
            ])
            .env(CHILD_CASE, case)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "initialization case {case} timed out (possible deadlock): {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "initialization case {case} failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed;"),
            "child case {case} did not execute exactly one test"
        );
        println!("initialization case {case}: passed");
    }
}
