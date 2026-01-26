//! List all keys in a safetensors file
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

fn main() {
    let device = Device::Cpu;
    
    // Check multiple models
    let models_to_check = vec![
        ("Intent Classifier", "../models/lora_intent_classifier_bert-base-uncased_model/model.safetensors"),
        ("PII Detector", "../models/lora_pii_detector_bert-base-uncased_model/model.safetensors"),
        ("Jailbreak Classifier", "../models/lora_jailbreak_classifier_bert-base-uncased_model/model.safetensors"),
    ];
    
    for (model_name, lora_path) in models_to_check {
        println!("\n{}", "=".repeat(60));
        println!("Checking: {}", model_name);
        println!("{}", "=".repeat(60));
        
        if !Path::new(lora_path).exists() {
            println!("File not found: {}", lora_path);
            continue;
        }
        
        println!("Loading safetensors file: {}", lora_path);
        let vb = unsafe {
            match VarBuilder::from_mmaped_safetensors(&[lora_path.to_string()], DType::F32, &device) {
                Ok(vb) => vb,
                Err(e) => {
                    eprintln!("Failed to load safetensors: {}", e);
                    continue;
                }
            }
        };
        
        // Try common patterns
        println!("\nChecking for LoRA weight keys...");
        let patterns = vec![
            "lora_intent.lora_A.weight",
            "lora_intent.lora_B.weight",
            "lora_pii.lora_A.weight",
            "lora_pii.lora_B.weight",
            "lora_jailbreak.lora_A.weight",
            "lora_jailbreak.lora_B.weight",
            "intent.lora_A.weight",
            "intent.lora_B.weight",
            "pii.lora_A.weight",
            "pii.lora_B.weight",
            "jailbreak.lora_A.weight",
            "jailbreak.lora_B.weight",
            "lora_A.weight",
            "lora_B.weight",
            "intent_classifier.weight",
            "intent_classifier.bias",
            "pii_classifier.weight",
            "pii_classifier.bias",
            "jailbreak_classifier.weight",
            "jailbreak_classifier.bias",
            "classifier.weight",
            "classifier.bias",
            "bert.classifier.weight",
            "bert.classifier.bias",
        ];
        
        let mut found_keys = Vec::new();
        for pattern in patterns {
            // Try to get shape first
            let mut found = false;
            match vb.get((1, 1), pattern) {
                Ok(t) => {
                    let shape = t.shape();
                    let dims: Vec<usize> = shape.dims().to_vec();
                    found_keys.push((pattern.to_string(), dims));
                    found = true;
                }
                Err(_) => {
                    // Try different shapes
                    let shapes = vec![(768, 768), (3, 768), (18, 768), (2, 768), (35, 768), (768, 1), (1, 768)];
                    for (h, w) in shapes {
                        if let Ok(t) = vb.get((h, w), pattern) {
                            let shape = t.shape();
                            let dims: Vec<usize> = shape.dims().to_vec();
                            found_keys.push((pattern.to_string(), dims));
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                println!("  ✗ Not found: {}", pattern);
            }
        }
        
        if found_keys.is_empty() {
            println!("  ⚠️  No matching keys found");
        } else {
            println!("  ✅ Found keys:");
            for (key, shape) in found_keys {
                println!("     - {} (shape: {:?})", key, shape);
            }
        }
    }
}
