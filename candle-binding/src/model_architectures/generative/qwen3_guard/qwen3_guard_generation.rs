use super::qwen3_guard_sampling::{apply_repeat_penalty, sample_argmax, sample_topp};
use super::Qwen3GuardModel;
use crate::core::{ConfigErrorType, UnifiedError, UnifiedResult};
use candle_core::{DType, Tensor};

impl Qwen3GuardModel {
    /// Generate with prefix caching (faster tokenization + KV reuse).
    pub(super) fn generate_with_prefix_cache(
        &mut self,
        text: &str,
        mode: &str,
    ) -> UnifiedResult<String> {
        let (prefix_tokens, prefix_len) = self.cached_prefix_snapshot(mode)?;

        self.model.clear_kv_cache();
        self.model
            .process_prefix(&prefix_tokens)
            .map_err(|e| UnifiedError::Processing {
                operation: "process prefix cache".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        let suffix = cached_guard_suffix(text, mode);
        let encoding = self.tokenizer.encode(suffix.as_str(), true).map_err(|e| {
            UnifiedError::Configuration {
                operation: "tokenize suffix".to_string(),
                source: ConfigErrorType::ParseError(e.to_string()),
                context: None,
            }
        })?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        self.process_cached_suffix(&tokens, prefix_len)?;
        self.generate_cached_suffix_tokens(tokens, prefix_len)
    }

    fn cached_prefix_snapshot(&self, mode: &str) -> UnifiedResult<(Vec<u32>, usize)> {
        let cache = match mode {
            "input" => self
                .prefix_cache_input
                .as_ref()
                .ok_or_else(|| prefix_cache_error("input cache not initialized"))?,
            "output" => self
                .prefix_cache_output
                .as_ref()
                .ok_or_else(|| prefix_cache_error("output cache not initialized"))?,
            _ => return Err(prefix_cache_error(&format!("invalid mode: {}", mode))),
        };
        Ok((cache.prefix_tokens().to_vec(), cache.prefix_length()))
    }

    fn process_cached_suffix(&mut self, tokens: &[u32], prefix_len: usize) -> UnifiedResult<()> {
        let suffix_tensor = Tensor::new(tokens, &self.device)
            .map_err(|e| UnifiedError::Processing {
                operation: "create suffix tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .unsqueeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "unsqueeze suffix tensor".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        self.model
            .forward(&suffix_tensor, prefix_len)
            .map_err(|e| UnifiedError::Processing {
                operation: "forward suffix".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;
        Ok(())
    }

    fn generate_cached_suffix_tokens(
        &mut self,
        mut tokens: Vec<u32>,
        prefix_len: usize,
    ) -> UnifiedResult<String> {
        let mut generated_text = String::new();
        let mut total_tokens = prefix_len + tokens.len();

        for _step in 0..self.config.max_tokens {
            let context_size = 1;
            let start_pos = total_tokens - context_size;
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)
                .map_err(|e| UnifiedError::Processing {
                    operation: "create tensor".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?
                .unsqueeze(0)
                .map_err(|e| UnifiedError::Processing {
                    operation: "unsqueeze".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;
            let logits = self.last_token_logits(&input, start_pos)?;
            let logits = if self.config.repeat_penalty != 1.0 {
                let start_at = tokens.len().saturating_sub(self.config.repeat_last_n);
                apply_repeat_penalty(&logits, self.config.repeat_penalty, &tokens[start_at..])?
            } else {
                logits
            };
            let next_token = self.sample_next_token(&logits)?;
            if next_token == self.eos_token_id || next_token == self.im_end_token_id {
                break;
            }

            tokens.push(next_token);
            total_tokens += 1;
            if let Ok(piece) = self.tokenizer.decode(&[next_token], true) {
                generated_text.push_str(&piece);
            }
        }

        Ok(generated_text)
    }

    fn last_token_logits(&mut self, input: &Tensor, start_pos: usize) -> UnifiedResult<Tensor> {
        self.model
            .forward(input, start_pos)
            .map_err(|e| UnifiedError::Processing {
                operation: "forward pass".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .squeeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "squeeze".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .squeeze(0)
            .map_err(|e| UnifiedError::Processing {
                operation: "squeeze".to_string(),
                source: e.to_string(),
                input_context: None,
            })?
            .to_dtype(DType::F32)
            .map_err(|e| UnifiedError::Processing {
                operation: "to_dtype".to_string(),
                source: e.to_string(),
                input_context: None,
            })
    }

    fn sample_next_token(&self, logits: &Tensor) -> UnifiedResult<u32> {
        if self.config.temperature == 0.0 {
            sample_argmax(logits)
        } else {
            sample_topp(logits, self.config.temperature, self.config.top_p)
        }
    }
}

fn prefix_cache_error(source: &str) -> UnifiedError {
    UnifiedError::Processing {
        operation: "get prefix cache".to_string(),
        source: source.to_string(),
        input_context: None,
    }
}

fn cached_guard_suffix(text: &str, mode: &str) -> String {
    let user_label = if mode == "output" {
        "ASSISTANT"
    } else {
        "USER"
    };
    format!(
        "{}\n\n\
        <END CONVERSATION>\n\n\
        Provide your safety assessment for ONLY THE LAST **{}'s query** in the above conversation:\n \
        - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.\n \
        - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>\n\
        <|im_start|>assistant\n\
        <think>\n\n\
        </think>\n\n",
        text, user_label
    )
}
