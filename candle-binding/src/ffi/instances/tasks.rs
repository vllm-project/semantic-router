use super::*;
use crate::core::device::run_on_inference_pool;
use crate::ffi::embedding::truncate_embedding_to_dimension;
use candle_core::{DType, Tensor};

#[derive(Default, Serialize)]
pub(super) struct InputMetadata {
    pub(super) input_tokens: usize,
    pub(super) processed_tokens: usize,
    pub(super) truncated: bool,
}

#[derive(Serialize)]
pub(super) struct Distribution {
    class: usize,
    confidence: f32,
    probabilities: Vec<f32>,
    labels: Vec<String>,
    input: InputMetadata,
}

#[derive(Serialize)]
pub(super) struct Span {
    text: String,
    start: usize,
    end: usize,
    confidence: f32,
    label: String,
}

#[derive(Serialize)]
pub(super) struct TokenOutput {
    spans: Vec<Span>,
    input: InputMetadata,
    offset_unit: &'static str,
}

#[derive(Serialize)]
pub(super) struct HallucinationOutput {
    has_hallucination: bool,
    confidence: f32,
    spans: Vec<Span>,
    input: InputMetadata,
    offset_unit: &'static str,
}

#[derive(Serialize)]
pub(super) struct EmbeddingOutput {
    values: Vec<f32>,
    input: InputMetadata,
}

impl Instance {
    fn tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("capability: headless backbone has no task tokenizer"))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self
            .tokenizer()?
            .encode(text, true)
            .map_err(|e| anyhow!(e.to_string()))?
            .len())
    }

    fn prepare_text<'a>(&self, text: &'a str) -> Result<(&'a str, InputMetadata)> {
        let count = self.count_tokens(text)?;
        let max = self.info.max_input_tokens;
        if count <= max {
            return Ok((
                text,
                InputMetadata {
                    input_tokens: count,
                    processed_tokens: count,
                    truncated: false,
                },
            ));
        }
        ensure!(
            self.info.overflow != "reject",
            "input_limit: input has {count} tokens, task budget is {max}"
        );
        let mut tokenizer = self.tokenizer()?.clone();
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max,
                ..Default::default()
            }))
            .map_err(|e| anyhow!(e.to_string()))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!(e.to_string()))?;
        let end = encoding
            .get_offsets()
            .iter()
            .map(|(_, end)| *end)
            .max()
            .unwrap_or(0);
        let mut selected = text
            .get(..end)
            .ok_or_else(|| anyhow!("result_invalid: invalid tokenizer offset"))?;
        // A prefix can retokenize differently at its boundary. Verify the actual
        // text passed to the maintained native classifier still fits its budget.
        while self.count_tokens(selected)? > max {
            let end = selected
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow!("input_limit: budget cannot hold special tokens"))?;
            selected = &selected[..end];
        }
        Ok((
            selected,
            InputMetadata {
                input_tokens: count,
                processed_tokens: self.count_tokens(selected)?,
                truncated: true,
            },
        ))
    }

    fn sequence_on_text(&self, text: &str, input: InputMetadata) -> Result<Distribution> {
        let (class, confidence, probabilities) = match &self.model {
            Model::Sequence(model) => model.classify_text_with_probabilities(text)?,
            Model::Bert(model) => model.classify_text_with_probabilities(text)?,
            Model::MergedBert(model) => model.classify_text_with_probabilities(text)?,
            Model::Deberta(model) => model.classify_text_with_probabilities(text)?,
            _ => bail!("capability: handle is not a sequence classifier"),
        };
        ensure!(
            !probabilities.is_empty()
                && probabilities
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
            "result_invalid: invalid probability values"
        );
        ensure!(
            (probabilities.iter().sum::<f32>() - 1.0).abs() <= 1e-4,
            "result_invalid: probability distribution does not sum to one"
        );
        ensure!(
            self.info.labels.len() == probabilities.len(),
            "result_invalid: label distribution shape mismatch"
        );
        Ok(Distribution {
            class,
            confidence,
            probabilities,
            labels: self.info.labels.clone(),
            input,
        })
    }

    pub(super) fn sequence(&self, text: &str) -> Result<Distribution> {
        ensure!(
            self.info.task == "sequence",
            "capability: wrong task handle"
        );
        let (text, input) = self.prepare_text(text)?;
        self.sequence_on_text(text, input)
    }

    pub(super) fn nli(&self, premise: &str, hypothesis: &str) -> Result<Distribution> {
        ensure!(self.info.task == "nli", "capability: wrong task handle");
        let Model::Sequence(model) = &self.model else {
            bail!("capability: NLI requires a ModernBERT classifier")
        };
        let tail = format!(" [SEP] {hypothesis}");
        ensure!(
            self.count_tokens(&tail)? <= self.info.max_input_tokens,
            "input_limit: hypothesis exceeds the NLI budget"
        );
        let original = format!("{premise}{tail}");
        if self.info.overflow == "reject" {
            let (text, input) = self.prepare_text(&original)?;
            return self.sequence_on_text(text, input);
        }
        let mut prefix = model.fit_prefix_to_window(premise, &tail)?;
        self.fit_pair_prefix(&mut prefix, &tail)?;
        let text = format!("{prefix}{tail}");
        let input = InputMetadata {
            input_tokens: self.count_tokens(&original)?,
            processed_tokens: self.count_tokens(&text)?,
            truncated: prefix != premise,
        };
        self.sequence_on_text(&text, input)
    }

    fn fit_pair_prefix(&self, prefix: &mut String, tail: &str) -> Result<()> {
        // Preserve the maintained prefix-window behavior, then account for a
        // configured smaller budget and tokenization across the segment boundary.
        while self.count_tokens(&format!("{prefix}{tail}"))? > self.info.max_input_tokens {
            let end = prefix
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .ok_or_else(|| anyhow!("input_limit: task suffix cannot fit"))?;
            prefix.truncate(end);
        }
        Ok(())
    }

    pub(super) fn tokens(&self, text: &str) -> Result<TokenOutput> {
        ensure!(self.info.task == "token", "capability: wrong task handle");
        let (text, input) = self.prepare_text(text)?;
        let predictions = match &self.model {
            Model::Token(model) => model.classify_tokens(text)?,
            Model::BertToken(model) => model.classify_tokens_with_offsets(text)?,
            Model::MergedBertToken(model) => model.classify_tokens_with_offsets(text)?,
            Model::LoRAToken(model) => model
                .classify_tokens(text)?
                .into_iter()
                .map(|token| {
                    (
                        token.token,
                        token.label_id,
                        token.confidence,
                        token.start_pos,
                        token.end_pos,
                    )
                })
                .collect(),
            _ => bail!("capability: handle is not a token classifier"),
        };
        let spans = predictions
            .into_iter()
            .filter(|(_, _, _, start, end)| end > start)
            .map(|(_, class, confidence, start, end)| {
                let text = text
                    .get(start..end)
                    .ok_or_else(|| anyhow!("result_invalid: invalid UTF-8 token offsets"))?;
                let label = self
                    .info
                    .labels
                    .get(class)
                    .ok_or_else(|| anyhow!("result_invalid: token label missing"))?;
                Ok(Span {
                    text: text.to_owned(),
                    start,
                    end,
                    confidence,
                    label: label
                        .strip_prefix("B-")
                        .or_else(|| label.strip_prefix("I-"))
                        .unwrap_or(label)
                        .to_owned(),
                })
            })
            .collect::<Result<_>>()?;
        Ok(TokenOutput {
            spans,
            input,
            offset_unit: "utf8_bytes",
        })
    }

    pub(super) fn hallucination(
        &self,
        context: &str,
        question: &str,
        answer: &str,
        threshold: f32,
    ) -> Result<HallucinationOutput> {
        ensure!(
            self.info.task == "hallucination",
            "capability: wrong task handle"
        );
        let Model::Token(model) = &self.model else {
            bail!("capability: hallucination requires token classifier")
        };
        let tail = if question.is_empty() {
            format!(" [SEP] {answer}")
        } else {
            format!(" Question: {question} [SEP] {answer}")
        };
        ensure!(
            self.count_tokens(&tail)? <= self.info.max_input_tokens,
            "input_limit: question and answer exceed the hallucination budget"
        );
        let original = format!("{context}{tail}");
        if self.info.overflow == "reject" {
            self.prepare_text(&original)?;
        }
        let mut prefix = model.fit_prefix_to_window(context, &tail)?;
        self.fit_pair_prefix(&mut prefix, &tail)?;
        let text = format!("{prefix}{tail}");
        let answer_start = text.len() - answer.len();
        let threshold = if threshold > 0.0 && threshold <= 1.0 {
            threshold
        } else {
            0.5
        };
        let tokens = model.classify_tokens(&text)?;
        let mut spans: Vec<Span> = Vec::new();
        let mut current: Option<Span> = None;
        for (_, class, confidence, start, end) in tokens {
            if start < answer_start || end <= start {
                continue;
            }
            if class == 1 && confidence >= threshold {
                let start = start - answer_start;
                let end = end - answer_start;
                ensure!(
                    answer.get(start..end).is_some(),
                    "result_invalid: invalid hallucination offsets"
                );
                if let Some(span) = current.as_mut() {
                    span.end = end;
                    span.confidence = span.confidence.max(confidence);
                } else {
                    current = Some(Span {
                        text: String::new(),
                        start,
                        end,
                        confidence,
                        label: "HALLUCINATED".to_owned(),
                    });
                }
            } else if let Some(span) = current.take() {
                spans.push(span);
            }
        }
        if let Some(span) = current {
            spans.push(span);
        }
        for span in &mut spans {
            span.text = answer
                .get(span.start..span.end)
                .ok_or_else(|| anyhow!("result_invalid: invalid merged span"))?
                .to_owned();
        }
        // Preserve the maintained detector's aggregate score. This is not a
        // calibrated probability; token confidences remain the actual softmax.
        let confidence = if spans.is_empty() {
            1.0
        } else {
            spans.iter().map(|s| s.confidence).fold(0.0, f32::max)
        };
        Ok(HallucinationOutput {
            has_hallucination: !spans.is_empty(),
            confidence,
            spans,
            offset_unit: "utf8_bytes",
            input: InputMetadata {
                input_tokens: self.count_tokens(&original)?,
                processed_tokens: self.count_tokens(&text)?,
                truncated: prefix != context,
            },
        })
    }

    pub(super) fn embedding(
        &self,
        text: &str,
        dimension: usize,
        layer: usize,
    ) -> Result<EmbeddingOutput> {
        ensure!(
            self.info.task == "embedding",
            "capability: wrong task handle"
        );
        let (text, input) = self.prepare_text(text)?;
        if let Model::BertEmbedding(model) = &self.model {
            ensure!(
                layer == 0,
                "capability: BERT embedding does not support layer early exit"
            );
            let values = model
                .get_embedding(text, Some(self.info.max_input_tokens))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            ensure!(
                dimension <= values.len(),
                "capability: requested dimension exceeds output size"
            );
            return Ok(EmbeddingOutput {
                values: truncate_embedding_to_dimension(
                    values,
                    (dimension > 0).then_some(dimension),
                ),
                input,
            });
        }
        let Model::Embedding(factory) = &self.model else {
            bail!("capability: handle is not an embedding model")
        };
        let encoding = self
            .tokenizer()?
            .encode(text, true)
            .map_err(|e| anyhow!(e.to_string()))?;
        let dimension = (dimension > 0).then_some(dimension);
        let layer = (layer > 0).then_some(layer);
        let run = |device: &Device| -> Result<Vec<f32>> {
            run_on_inference_pool(device, || {
                let ids = Tensor::new(encoding.get_ids(), device)?.unsqueeze(0)?;
                let mask = Tensor::new(encoding.get_attention_mask(), device)?.unsqueeze(0)?;
                let output = if let Some(model) = factory.get_qwen3_model() {
                    ensure!(
                        layer.is_none(),
                        "capability: Qwen3 does not support layer early exit"
                    );
                    model.embedding_forward(&ids, &mask)?
                } else if let Some(model) = factory.get_gemma_model() {
                    ensure!(
                        layer.is_none(),
                        "capability: Gemma does not support layer early exit"
                    );
                    model.embedding_forward(&ids, Some(&mask))?
                } else if let Some(model) = factory.get_mmbert_model() {
                    model.embedding_forward_with_matryoshka(&ids, Some(&mask), layer, dimension)?
                } else if let Some(model) = factory.get_multimodal_model() {
                    model.encode_text_with_matryoshka(&ids, Some(&mask), layer, dimension)?
                } else {
                    bail!("capability: no embedding model")
                };
                let values = output.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                ensure!(
                    dimension.is_none_or(|d| d <= values.len()),
                    "capability: embedding dimension exceeds output size"
                );
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "result_invalid: embedding contains non-finite values"
                );
                Ok(truncate_embedding_to_dimension(values, dimension))
            })
        };
        let values = if let Some(model) = factory.get_qwen3_model() {
            run(&model.device())?
        } else if let Some(model) = factory.get_gemma_model() {
            run(&model.device())?
        } else if let Some(model) = factory.get_mmbert_model() {
            run(model.device())?
        } else if let Some(model) = factory.get_multimodal_model() {
            run(model.device())?
        } else {
            bail!("capability: no embedding model")
        };
        Ok(EmbeddingOutput { values, input })
    }
}

impl Instance {
    pub(super) fn image(&self, bytes: &[u8], dimension: usize) -> Result<EmbeddingOutput> {
        let Model::Embedding(factory) = &self.model else {
            bail!("capability: image embedding requires a multimodal model")
        };
        let model = factory
            .get_multimodal_model()
            .ok_or_else(|| anyhow!("capability: image embedding requires a multimodal model"))?;
        let pixels = crate::ffi::embedding::decode_resize_to_chw_f32(bytes, 512, 512)
            .map_err(|e| anyhow!(e))?;
        run_on_inference_pool(model.device(), || {
            let tensor = Tensor::from_slice(&pixels, (1, 3, 512, 512), model.device())?;
            let output =
                model.encode_image_with_dim(&tensor, (dimension > 0).then_some(dimension))?;
            let values = output.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            ensure!(
                dimension == 0 || dimension == values.len(),
                "capability: unsupported image embedding dimension"
            );
            Ok(EmbeddingOutput {
                values,
                input: InputMetadata::default(),
            })
        })
    }

    pub(super) fn audio(
        &self,
        samples: &[f32],
        mel_bins: usize,
        frames: usize,
        dimension: usize,
    ) -> Result<EmbeddingOutput> {
        let Model::Embedding(factory) = &self.model else {
            bail!("capability: audio embedding requires a multimodal model")
        };
        let model = factory
            .get_multimodal_model()
            .ok_or_else(|| anyhow!("capability: audio embedding requires a multimodal model"))?;
        ensure!(
            mel_bins > 0 && frames > 0 && mel_bins.checked_mul(frames) == Some(samples.len()),
            "configuration: invalid mel spectrogram shape"
        );
        ensure!(
            samples.iter().all(|sample| sample.is_finite()),
            "configuration: spectrogram values must be finite"
        );
        run_on_inference_pool(model.device(), || {
            let tensor = Tensor::from_slice(samples, (1, mel_bins, frames), model.device())?;
            let output =
                model.encode_audio_with_dim(&tensor, (dimension > 0).then_some(dimension))?;
            let values = output.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            ensure!(
                dimension == 0 || dimension == values.len(),
                "capability: unsupported audio embedding dimension"
            );
            Ok(EmbeddingOutput {
                values,
                input: InputMetadata::default(),
            })
        })
    }
}

impl Instance {
    pub(super) fn text_windows(
        &self,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<(usize, usize)>> {
        ensure!(
            self.info.task == "embedding",
            "capability: wrong task handle"
        );
        let limit = if max_tokens == 0 {
            self.info.max_input_tokens
        } else {
            max_tokens.min(self.info.max_input_tokens)
        };
        let specials = self.count_tokens("")?;
        ensure!(
            limit > specials,
            "input_limit: token window cannot contain model special tokens"
        );
        let encoded = self
            .tokenizer()?
            .encode(text, false)
            .map_err(|error| anyhow!(error.to_string()))?;
        let offsets: Vec<_> = encoded
            .get_offsets()
            .iter()
            .copied()
            .filter(|(start, end)| end > start)
            .collect();
        let budget = limit - specials;
        Ok(crate::core::tokenization_window::window_ranges(
            &offsets, budget, budget,
        ))
    }
}
