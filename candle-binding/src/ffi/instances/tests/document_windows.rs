use super::*;

#[test]
fn independent_document_budget_preserves_direct_and_shared_forward_limits() {
    let dir = fixture(&["safe", "unsafe"], 1);
    update_fixture_config(&dir, |config| {
        config["max_position_embeddings"] = json!(32768);
        // This tests admission and exact coverage, not attention quality or GPUs.
        config["num_hidden_layers"] = json!(0);
    });
    let mut opts = options(&dir);
    opts.overflow = "reject".into();
    opts.max_input_tokens = 32768;
    opts.document_max_input_tokens = 65536;
    let direct = load(opts.clone(), "sequence").unwrap();
    let backbone = load(opts.clone(), "backbone").unwrap();
    let bound = bind_head(&backbone, dir.path().to_str().unwrap(), "sequence").unwrap();
    let text = format!("{}秘密", "hello ".repeat(65533));
    for model in [&direct, &bound] {
        assert_eq!(model.info.max_input_tokens, 32768);
        assert_eq!(model.info.document_max_input_tokens, 65536);
        assert!(model.sequence(&text).is_err());
        let result = value(model.classify_windows(&text, 32768, 128).unwrap());
        assert_eq!(result["input"]["input_tokens"], 65536);
        assert_eq!(result["input"]["processed_tokens"], 65536);
        assert_eq!(result["input"]["truncated"], false);
        assert_eq!(result["content_tokens"], 65534);
        let windows = result["windows"].as_array().unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0]["start"], 0);
        assert_eq!(windows[0]["end"], 32766);
        assert_eq!(windows[1]["start"], 32638);
        assert_eq!(windows[2]["end"], 65534);
        assert!(model.classify_windows(&text, 32769, 128).is_err());
        assert!(model
            .classify_windows(&format!("{text} hello"), 32768, 128)
            .is_err());
    }
    opts.max_input_tokens = 32769;
    assert!(load(opts, "sequence").is_err());
}

#[test]
fn token_windows_scan_beyond_model_capacity_without_losing_tail_offsets() {
    let dir = fixture(&["O", "B-SECRET"], 1);
    update_fixture_config(&dir, |config| {
        config["max_position_embeddings"] = json!(8);
        config["num_hidden_layers"] = json!(0);
    });
    let mut opts = options(&dir);
    opts.overflow = "reject".into();
    opts.max_input_tokens = 8;
    opts.document_max_input_tokens = 24;
    let direct = load(opts.clone(), "token").unwrap();
    let backbone = load(opts, "backbone").unwrap();
    let bound = bind_head(&backbone, dir.path().to_str().unwrap(), "token").unwrap();
    let text = format!("{}猫", "hello ".repeat(21));
    for model in [&direct, &bound] {
        assert!(model.tokens(&text).is_err());
        let result = value(model.token_windows(&text, 8, 2).unwrap());
        assert_eq!(result["input"]["processed_tokens"], 24);
        assert_eq!(result["input"]["truncated"], false);
        assert_eq!(result["content_tokens"], 22);
        assert_eq!(
            result["windows"],
            json!([[0, 6], [4, 10], [8, 14], [12, 18], [16, 22]])
        );
        let tail = result["spans"].as_array().unwrap().last().unwrap();
        assert_eq!(tail["text"], "猫");
        assert_eq!(tail["start"], text.len() - "猫".len());
        assert_eq!(tail["end"], text.len());
        assert!(model.token_windows(&text, 9, 2).is_err());
        assert!(model.token_windows(&format!("{text} hello"), 8, 2).is_err());
    }
}

#[test]
fn label_windows_use_document_budget_even_with_single_forward_truncation() {
    let dir = fixture(&["first", "second"], 1);
    update_fixture_config(&dir, |config| {
        config["problem_type"] = json!("multi_label_classification");
        config["max_position_embeddings"] = json!(8);
    });
    let mut opts = options(&dir);
    opts.max_input_tokens = 8;
    opts.document_max_input_tokens = 24;
    let model = load(opts.clone(), "label_scores").unwrap();
    let text = "hello ".repeat(22);
    assert_eq!(
        value(model.score(&text).unwrap())["input"]["processed_tokens"],
        8
    );
    let output = value(model.score_windows(&text, 8, 2).unwrap());
    assert_eq!(output["input"]["processed_tokens"], 24);
    assert_eq!(output["input"]["truncated"], false);
    assert_eq!(output["windows"].as_array().unwrap().len(), 5);
    assert!(model.score_windows(&format!("{text}hello"), 8, 2).is_err());
    opts.document_max_input_tokens = 7;
    assert!(load(opts.clone(), "label_scores").is_err());
    opts.document_max_input_tokens = 24;
    assert!(load(opts, "embedding").is_err());
}
