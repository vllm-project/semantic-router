//! Shared ModernBERT checkpoint field validation.

use candle_core::Result;
use serde_json::Value;

/// Official `norm_eps` is canonical. Preserve the legacy Candle spelling only
/// when unambiguous. Absence is left to the caller's existing default policy.
pub(crate) fn norm_eps(raw: &Value) -> Result<Option<f64>> {
    let read = |name: &str| -> Result<Option<f64>> {
        raw.get(name)
            .map(|value| match value.as_f64() {
                Some(value) if value.is_finite() && value > 0.0 => Ok(value),
                _ => candle_core::bail!("ModernBERT {name} must be finite and positive"),
            })
            .transpose()
    };
    let official = read("norm_eps")?;
    let legacy = read("layer_norm_eps")?;
    if official.zip(legacy).is_some_and(|(a, b)| a != b) {
        candle_core::bail!("conflicting ModernBERT norm_eps and legacy layer_norm_eps");
    }
    Ok(official.or(legacy))
}
