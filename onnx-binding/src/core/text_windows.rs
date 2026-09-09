//! Tokenizer-owned byte ranges for embedding a complete RAG query.

use tokenizers::{PostProcessor, Tokenizer};

pub fn text_windows(
    tokenizer: &Tokenizer,
    text: &str,
    model_window: usize,
    max_length: Option<usize>,
) -> tokenizers::Result<Vec<(usize, usize)>> {
    let window = tokenizer
        .get_truncation()
        .map_or(model_window, |params| model_window.min(params.max_length));
    let window = max_length.map_or(window, |limit| window.min(limit));
    let special_tokens = tokenizer
        .get_post_processor()
        .map_or(0, |processor| processor.added_tokens(false));
    let budget = window
        .checked_sub(special_tokens)
        .filter(|budget| *budget > 0)
        .ok_or("embedding window has no room for text after special tokens")?;

    let mut tokenizer = tokenizer.clone();
    tokenizer.with_truncation(None)?;
    tokenizer.with_padding(None);
    let encoding = tokenizer.encode(text, false)?;
    let offsets: Vec<_> = encoding
        .get_offsets()
        .iter()
        .copied()
        .filter(|(start, end)| end > start)
        .collect();
    if offsets.is_empty() {
        return Ok(Vec::new());
    }
    if tokenizer.encode(text, true)?.len() <= window {
        return Ok(vec![(0, text.len())]);
    }

    let mut ranges = Vec::new();
    let mut start = 0;
    while start < offsets.len() {
        let mut end = start.saturating_add(budget).min(offsets.len());
        loop {
            if end == start {
                return Err("embedding window cannot fit one text boundary".into());
            }
            let range = (offsets[start].0, offsets[end - 1].1);
            let slice = text
                .get(range.0..range.1)
                .ok_or("tokenizer returned invalid UTF-8 byte boundaries")?;
            // A substring can tokenize differently at its new leading boundary.
            if tokenizer.encode(slice, true)?.len() <= window {
                ranges.push(range);
                break;
            }
            end -= 1;
        }
        if end == offsets.len() {
            break;
        }
        start += (end - start).div_ceil(2);
    }
    Ok(ranges)
}

#[cfg(test)]
#[path = "text_windows_test.rs"]
mod tests;
