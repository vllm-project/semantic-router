use crate::core::{UnifiedError, UnifiedResult};
use candle_core::Tensor;

pub(super) fn apply_repeat_penalty(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
) -> UnifiedResult<Tensor> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    let mut modified_logits = logits_vec.clone();
    for &token_id in context {
        let idx = token_id as usize;
        if idx < modified_logits.len() {
            if modified_logits[idx] < 0.0 {
                modified_logits[idx] *= penalty;
            } else {
                modified_logits[idx] /= penalty;
            }
        }
    }

    Tensor::new(modified_logits, logits.device()).map_err(|e| UnifiedError::Processing {
        operation: "create tensor".to_string(),
        source: e.to_string(),
        input_context: None,
    })
}

pub(super) fn sample_argmax(logits: &Tensor) -> UnifiedResult<u32> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    let max_idx = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(max_idx as u32)
}

pub(super) fn sample_topp(logits: &Tensor, temperature: f64, top_p: f64) -> UnifiedResult<u32> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| UnifiedError::Processing {
            operation: "to_vec1".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

    let scaled_logits: Vec<f32> = logits_vec.iter().map(|&x| x / temperature as f32).collect();
    let max_logit = scaled_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = scaled_logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative_prob = 0.0;
    let mut sampled_probs = Vec::new();
    for (idx, prob) in indexed_probs.iter() {
        sampled_probs.push((*idx, *prob));
        cumulative_prob += prob;
        if cumulative_prob >= top_p as f32 {
            break;
        }
    }

    let filtered_sum: f32 = sampled_probs.iter().map(|(_, p)| p).sum();
    let normalized_probs: Vec<(usize, f32)> = sampled_probs
        .iter()
        .map(|(idx, p)| (*idx, p / filtered_sum))
        .collect();

    use rand::Rng;
    let mut rng = rand::rng();
    let random_value: f32 = rng.random();

    let mut cumulative = 0.0;
    for (idx, prob) in normalized_probs.iter() {
        cumulative += prob;
        if random_value <= cumulative {
            return Ok(*idx as u32);
        }
    }

    Ok(normalized_probs
        .last()
        .map(|(idx, _)| *idx as u32)
        .unwrap_or(0))
}
