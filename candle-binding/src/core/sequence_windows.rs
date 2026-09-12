//! Exact token windows. Keep the Candle and ONNX implementations equivalent.
//! No decode/re-encode step may alter tokens at a window boundary.

pub struct TokenWindow {
    pub start: usize,
    pub end: usize,
    pub ids: Vec<u32>,
}

pub fn encode_windows(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    input_limit: usize,
    size: usize,
    overlap: usize,
) -> Result<Vec<TokenWindow>, String> {
    let content = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
    let full = tokenizer.encode(text, true).map_err(|e| e.to_string())?;
    if full.len() > input_limit {
        return Err(format!(
            "input has {} tokens, exceeds {input_limit} token model budget",
            full.len()
        ));
    }
    if size > input_limit {
        return Err("window size exceeds the model input budget".into());
    }
    let mask = full.get_special_tokens_mask();
    let first = mask
        .iter()
        .position(|&v| v == 0)
        .ok_or("window scanning requires nonempty content")?;
    let last = mask.iter().rposition(|&v| v == 0).unwrap() + 1;
    if mask[first..last].iter().any(|&v| v != 0)
        || full.get_ids()[first..last] != *content.get_ids()
    {
        return Err("window scanning requires contiguous content between special tokens".into());
    }
    let prefix = &full.get_ids()[..first];
    let suffix = &full.get_ids()[last..];
    let empty = tokenizer.encode("", true).map_err(|e| e.to_string())?;
    if [prefix, suffix].concat() != empty.get_ids() {
        return Err("window scanning requires a fixed special-token prefix and suffix".into());
    }
    token_windows(content.get_ids(), prefix, suffix, size, overlap)
}

fn token_windows(
    content: &[u32],
    prefix: &[u32],
    suffix: &[u32],
    size: usize,
    overlap: usize,
) -> Result<Vec<TokenWindow>, String> {
    let width = size
        .checked_sub(prefix.len() + suffix.len())
        .filter(|&n| n > 0)
        .ok_or("window size must leave room for content after special tokens")?;
    if overlap >= width {
        return Err("window overlap must be smaller than its content width".into());
    }
    if content.is_empty() {
        return Err("window scanning requires nonempty content".into());
    }
    let stride = width - overlap;
    let mut windows = Vec::new();
    let mut start = 0;
    while start < content.len() {
        let end = start.saturating_add(width).min(content.len());
        let ids = [prefix, &content[start..end], suffix].concat();
        windows.push(TokenWindow { start, end, ids });
        if end == content.len() {
            break;
        }
        start += stride;
    }
    Ok(windows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_windows_cover_boundaries_and_restore_specials() {
        for length in [1, 509, 510, 511, 512, 765, 766, 32766] {
            // Literal special-token IDs in content must survive unchanged.
            let content: Vec<u32> = (0..length).map(|n| (n % 29) as u32).collect();
            let windows = token_windows(&content, &[2], &[1], 512, 255).unwrap();
            let mut covered = vec![false; length];
            for (index, window) in windows.iter().enumerate() {
                assert_eq!(window.start, index * 255);
                assert_eq!(
                    window.ids,
                    [&[2][..], &content[window.start..window.end], &[1][..]].concat()
                );
                assert!(window.ids.len() <= 512);
                covered[window.start..window.end].fill(true);
                if index + 1 < windows.len() {
                    assert!(window.end < length);
                }
            }
            assert!(covered.iter().all(|&v| v));
            assert_eq!(windows.last().unwrap().end, length);
        }
        let windows = token_windows(&[7; 766], &[2], &[1], 512, 255).unwrap();
        assert_eq!(
            windows.iter().map(|w| (w.start, w.end)).collect::<Vec<_>>(),
            vec![(0, 510), (255, 765), (510, 766)]
        );
    }

    #[test]
    fn invalid_window_budgets_are_rejected_without_clamping() {
        assert!(token_windows(&[], &[2], &[1], 512, 255).is_err());
        assert!(token_windows(&[7], &[2], &[1], 2, 0).is_err());
        assert!(token_windows(&[7], &[2], &[1], 512, 510).is_err());
        assert_eq!(token_windows(&[7; 7], &[2], &[1], 5, 0).unwrap().len(), 3);
    }

    #[test]
    fn tokenizer_windows_match_whole_input_and_reject_overflow() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::processors::bert::BertProcessing;
        let vocabulary = ["[UNK]", "[SEP]", "[CLS]", "word", "tail"]
            .iter()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), 1),
            ("[CLS]".into(), 2),
        )));
        let short = "word tail";
        let windows = encode_windows(&tokenizer, short, 32768, 512, 255).unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(
            windows[0].ids,
            tokenizer.encode(short, true).unwrap().get_ids()
        );
        let long = "word ".repeat(32765) + "tail";
        let windows = encode_windows(&tokenizer, &long, 32768, 512, 255).unwrap();
        assert_eq!(windows.last().unwrap().end, 32766);
        assert_eq!(
            windows.last().unwrap().ids[windows.last().unwrap().ids.len() - 2],
            4
        );
        assert!(encode_windows(&tokenizer, &(long + " word"), 32768, 512, 255).is_err());
        assert!(encode_windows(&tokenizer, "", 32768, 512, 255).is_err());
        assert!(encode_windows(&tokenizer, short, 512, 513, 0).is_err());
        assert!(encode_windows(&tokenizer, short, 512, 512, 510).is_err());
    }
}
