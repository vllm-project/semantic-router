use anyhow::Result;

use super::tokenization::DualPathTokenizer;

pub fn fit_prefix_to_window(
    tokenizer: &dyn DualPathTokenizer,
    prefix: &str,
    suffix: &str,
) -> Result<String> {
    let max_length = tokenizer.get_config().max_length;
    let suffix_tokens = tokenizer.tokenize(suffix)?.token_ids.len();
    let budget = max_length.saturating_sub(suffix_tokens);
    let content: Vec<(usize, usize)> = tokenizer
        .tokenize(prefix)?
        .offsets
        .into_iter()
        .filter(|(start, end)| end > start)
        .collect();
    if content.len() <= budget {
        return Ok(prefix.to_string());
    }
    let end = if budget == 0 {
        0
    } else {
        content[budget - 1].1
    };
    Ok(prefix[..end].to_string())
}
