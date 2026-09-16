use candle_semantic_router::{
    classify_text, classify_text_with_probabilities, free_probabilities, init_generic_classifier,
};
use std::ffi::CString;

#[test]
#[ignore = "requires a local checkpoint; set CANDLE_GENERIC_CLASSIFIER_MODEL and CANDLE_GENERIC_CLASSIFIER_NUM_CLASSES"]
fn generic_classifier_checkpoint_loads_and_classifies_on_cpu() {
    let path = std::env::var("CANDLE_GENERIC_CLASSIFIER_MODEL")
        .expect("set CANDLE_GENERIC_CLASSIFIER_MODEL to a local checkpoint");
    let num_classes: i32 = std::env::var("CANDLE_GENERIC_CLASSIFIER_NUM_CLASSES")
        .expect("set CANDLE_GENERIC_CLASSIFIER_NUM_CLASSES")
        .parse()
        .expect("class count must be an integer");
    let model_path = CString::new(path.as_str()).unwrap();
    println!("model={path} num_classes={num_classes} use_cpu=true");
    assert!(
        init_generic_classifier(model_path.as_ptr(), num_classes, true),
        "generic classifier initialization failed"
    );
    for text in [
        "Please explain how solar panels produce electricity.",
        "The meeting starts at nine tomorrow.",
    ] {
        let input = CString::new(text).unwrap();
        let top = classify_text(input.as_ptr());
        let result = classify_text_with_probabilities(input.as_ptr());
        assert!(result.predicted_class >= 0);
        assert_eq!(result.num_classes, num_classes);
        assert!(!result.probabilities.is_null());
        let probabilities = unsafe {
            std::slice::from_raw_parts(result.probabilities, result.num_classes as usize).to_vec()
        };
        free_probabilities(result.probabilities, result.num_classes);
        assert!(result.label.is_null());
        assert!(probabilities
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(top.predicted_class, result.predicted_class);
        assert_eq!(top.confidence, result.confidence);
        assert_eq!(
            probabilities[result.predicted_class as usize],
            result.confidence
        );
        println!(
            "input={text:?} class={} confidence={} probabilities={probabilities:?}",
            result.predicted_class, result.confidence
        );
    }
}
