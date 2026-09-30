//! Config-driven ModernBERT rotary embeddings shared by both encoder paths.
//!
//! Default RoPE preserves the original Candle FP32 arithmetic. Standard YaRN
//! follows Transformers' frequency interpolation and attention scaling; changing
//! theta alone does not enable YaRN. Caches belong to their model/device/dtype.

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct RopeOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scaling {
    Default,
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        beta_fast: f64,
        beta_slow: f64,
        attention_factor: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    pub theta: f64,
    pub scaling: Scaling,
}

impl Parameters {
    pub fn unscaled(theta: f64) -> Self {
        Self {
            theta,
            scaling: Scaling::Default,
        }
    }

    pub fn frequencies(&self, dim: usize) -> Result<(Vec<f32>, f64)> {
        if dim == 0 || !dim.is_multiple_of(2) || !self.theta.is_finite() || self.theta <= 0.0 {
            candle_core::bail!("ModernBERT RoPE requires an even positive head dimension and positive finite theta");
        }
        match self.scaling {
            Scaling::Default => {
                // Keep the old f64 power -> f32 denominator -> f32 reciprocal.
                let frequencies = (0..dim)
                    .step_by(2)
                    .map(|i| 1f32 / self.theta.powf(i as f64 / dim as f64) as f32)
                    .collect();
                Ok((frequencies, 1.0))
            }
            Scaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                attention_factor,
            } => {
                let correction = |rotations: f64| {
                    dim as f64
                        * (original_max_position_embeddings as f64
                            / (rotations * 2.0 * std::f64::consts::PI))
                            .ln()
                        / (2.0 * self.theta.ln())
                };
                let low = correction(beta_fast).floor().max(0.0);
                let mut high = correction(beta_slow).ceil().min((dim - 1) as f64);
                if low == high {
                    high += 0.001;
                }
                let frequencies = (0..dim)
                    .step_by(2)
                    .enumerate()
                    .map(|(index, i)| {
                        let power = (self.theta as f32).powf(i as f32 / dim as f32);
                        let extrapolation = 1.0f32 / power;
                        let interpolation = 1.0f32 / (factor * power);
                        let ramp =
                            ((index as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                        let extrapolation_factor = 1.0f32 - ramp;
                        interpolation * (1.0f32 - extrapolation_factor)
                            + extrapolation * extrapolation_factor
                    })
                    .collect();
                Ok((frequencies, attention_factor))
            }
        }
    }
}

fn positive(value: &Value, field: &str) -> Result<f64> {
    match value.as_f64() {
        Some(number) if number.is_finite() && number > 0.0 => Ok(number),
        _ => candle_core::bail!("ModernBERT RoPE {field} must be finite and positive"),
    }
}

/// Resolve theta without interpreting variant names or training sidecars.
/// `defaults` preserve the calling loader's existing legacy missing-field policy.
pub fn resolve_thetas(raw: &Value, defaults: [f64; 2]) -> Result<[f64; 2]> {
    let mut result = defaults;
    for (index, (field, layer)) in [
        ("global_rope_theta", "full_attention"),
        ("local_rope_theta", "sliding_attention"),
    ]
    .iter()
    .enumerate()
    {
        let legacy = raw
            .get(*field)
            .filter(|value| !value.is_null())
            .map(|value| positive(value, field))
            .transpose()?;
        let nested = raw
            .get("rope_parameters")
            .and_then(|params| params.get(*layer))
            .and_then(|params| params.get("rope_theta"))
            .map(|value| positive(value, "rope_theta"))
            .transpose()?;
        if legacy.zip(nested).is_some_and(|(old, new)| old != new) {
            candle_core::bail!("conflicting ModernBERT {field} and {layer}.rope_theta");
        }
        if raw
            .get("rope_parameters")
            .is_some_and(|value| !value.is_null())
            && legacy.is_none()
            && nested.is_none()
        {
            candle_core::bail!(
                "explicit ModernBERT rope_parameters requires an unambiguous {layer} theta"
            );
        }
        result[index] = nested.or(legacy).unwrap_or(defaults[index]);
        if !result[index].is_finite() || result[index] <= 0.0 {
            candle_core::bail!("ModernBERT RoPE theta must be finite and positive");
        }
    }
    Ok(result)
}

fn parse_parameters(value: Option<&Value>, theta: f64, max_positions: usize) -> Result<Parameters> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Parameters::unscaled(theta));
    };
    let map = value.as_object().ok_or_else(|| {
        candle_core::Error::Msg("ModernBERT RoPE parameters must be an object".into())
    })?;
    let kind = map.get("rope_type").or_else(|| map.get("type"));
    if map
        .get("rope_type")
        .zip(map.get("type"))
        .is_some_and(|(a, b)| a != b)
    {
        candle_core::bail!("conflicting ModernBERT rope_type and type");
    }
    let kind = match kind {
        None => "default",
        Some(value) => value.as_str().ok_or_else(|| {
            candle_core::Error::Msg("ModernBERT rope_type must be a string".into())
        })?,
    };
    for key in map.keys() {
        let common = matches!(key.as_str(), "rope_type" | "type" | "rope_theta");
        let yarn = kind == "yarn"
            && matches!(
                key.as_str(),
                "factor"
                    | "original_max_position_embeddings"
                    | "beta_fast"
                    | "beta_slow"
                    | "attention_factor"
                    | "truncate"
            );
        if !common && !yarn {
            candle_core::bail!("unsupported ModernBERT {kind} RoPE option {key}");
        }
    }
    if let Some(value) = map.get("rope_theta") {
        if positive(value, "rope_theta")? != theta {
            candle_core::bail!("conflicting ModernBERT RoPE theta");
        }
    }
    if kind == "default" {
        return Ok(Parameters::unscaled(theta));
    }
    if kind != "yarn" {
        candle_core::bail!("unsupported ModernBERT RoPE type {kind}");
    }
    if theta <= 1.0 || !(theta as f32).is_finite() || theta as f32 <= 1.0 {
        candle_core::bail!(
            "ModernBERT YaRN theta must be greater than one and representable in FP32"
        );
    }
    if map
        .get("truncate")
        .is_some_and(|value| value.as_bool() != Some(true))
    {
        candle_core::bail!("ModernBERT YaRN supports only truncate=true");
    }
    let factor = positive(
        map.get("factor")
            .ok_or_else(|| candle_core::Error::Msg("ModernBERT YaRN factor is required".into()))?,
        "factor",
    )?;
    if factor < 1.0 || !(factor as f32).is_finite() {
        candle_core::bail!("ModernBERT YaRN factor must be >= 1 and representable in FP32");
    }
    let original = match map.get("original_max_position_embeddings") {
        None => max_positions,
        Some(value) => value
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    "ModernBERT YaRN original_max_position_embeddings must be a positive integer"
                        .into(),
                )
            })?,
    };
    let optional = |key: &str, default: f64| -> Result<f64> {
        map.get(key)
            .filter(|value| !value.is_null())
            .map(|value| positive(value, key))
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let beta_fast = optional("beta_fast", 32.0)?;
    let beta_slow = optional("beta_slow", 1.0)?;
    if beta_fast < beta_slow {
        candle_core::bail!("ModernBERT YaRN beta_fast must be >= beta_slow");
    }
    let attention_factor = optional("attention_factor", 1.0 + 0.1 * factor.ln())?;
    if !(attention_factor as f32).is_finite() || attention_factor as f32 <= 0.0 {
        candle_core::bail!("ModernBERT YaRN attention_factor must be representable in FP32");
    }
    Ok(Parameters {
        theta,
        scaling: Scaling::Yarn {
            factor: factor as f32,
            original_max_position_embeddings: original,
            beta_fast,
            beta_slow,
            attention_factor,
        },
    })
}

impl RopeOptions {
    pub fn resolve(
        &self,
        thetas: [f64; 2],
        max_positions: usize,
        hidden: usize,
        heads: usize,
        layers: usize,
        cadence: usize,
    ) -> Result<[Parameters; 2]> {
        if heads == 0
            || hidden == 0
            || !hidden.is_multiple_of(heads)
            || !(hidden / heads).is_multiple_of(2)
            || cadence == 0
            || max_positions == 0
            || u32::try_from(max_positions).is_err()
        {
            candle_core::bail!(
                "invalid ModernBERT RoPE dimensions, capacity, or attention cadence"
            );
        }
        if self
            .partial_rotary_factor
            .is_some_and(|factor| factor != 1.0)
        {
            candle_core::bail!("unsupported ModernBERT partial_rotary_factor (requires 1)");
        }
        if let Some(types) = &self.layer_types {
            if types.len() != layers
                || types.iter().enumerate().any(|(index, kind)| {
                    kind != if index % cadence == 0 {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                })
            {
                candle_core::bail!("unsupported ModernBERT layer_types: must match global_attn_every_n_layers cadence");
            }
        }
        let mut result = [
            Parameters::unscaled(thetas[0]),
            Parameters::unscaled(thetas[1]),
        ];
        if let Some(params) = &self.rope_parameters {
            let map = params.as_object().ok_or_else(|| {
                candle_core::Error::Msg(
                    "ModernBERT rope_parameters must be a per-layer object".into(),
                )
            })?;
            if map
                .keys()
                .any(|key| key != "full_attention" && key != "sliding_attention")
            {
                candle_core::bail!("unsupported ModernBERT rope_parameters key; expected full_attention/sliding_attention");
            }
            for (index, layer) in ["full_attention", "sliding_attention"].iter().enumerate() {
                let value = map
                    .get(*layer)
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "ModernBERT rope_parameters requires an explicit {layer} object"
                        ))
                    })?;
                result[index] = parse_parameters(Some(value), thetas[index], max_positions)?;
            }
        }
        if let Some(legacy) = &self.rope_scaling {
            for index in 0..2 {
                let parsed = parse_parameters(Some(legacy), thetas[index], max_positions)?;
                if self.rope_parameters.is_some() && result[index] != parsed {
                    candle_core::bail!("conflicting ModernBERT rope_scaling and rope_parameters");
                }
                result[index] = parsed;
            }
        }
        for params in &result {
            params.frequencies(hidden / heads)?;
        }
        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    pub(crate) sin: Tensor,
    pub(crate) cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        dim: usize,
        max_positions: usize,
        params: &Parameters,
        device: &Device,
    ) -> Result<Self> {
        if max_positions == 0 || u32::try_from(max_positions).is_err() {
            candle_core::bail!("ModernBERT RoPE capacity must be a positive u32");
        }
        let (frequencies, attention_factor) = params.frequencies(dim)?;
        let frequency_count = frequencies.len();
        let frequencies = Tensor::from_vec(frequencies, (1, frequency_count), device)?;
        // Preserve positions/angles in FP32, then scale sin/cos before the cast.
        let positions = Tensor::arange(0u32, max_positions as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_positions, 1))?;
        let angles = positions.matmul(&frequencies)?;
        let mut sin = angles.sin()?;
        let mut cos = angles.cos()?;
        if attention_factor != 1.0 {
            sin = (sin * attention_factor)?;
            cos = (cos * attention_factor)?;
        }
        Ok(Self {
            sin: sin.to_dtype(dtype)?,
            cos: cos.to_dtype(dtype)?,
        })
    }

    pub fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q, k))
    }
}

#[cfg(test)]
#[path = "modernbert_rope_test.rs"]
pub(crate) mod tests;
