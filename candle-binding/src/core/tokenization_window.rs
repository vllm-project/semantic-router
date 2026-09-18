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

/// Group token offsets into byte ranges of at most `budget` tokens each,
/// starting a new range every `stride` tokens.
///
/// A stride below the budget makes the ranges overlap, so a sentence that one
/// boundary cuts is whole inside the next range.
pub fn window_ranges(
    offsets: &[(usize, usize)],
    budget: usize,
    stride: usize,
) -> Vec<(usize, usize)> {
    if offsets.is_empty() || budget == 0 {
        return Vec::new();
    }
    let stride = stride.clamp(1, budget);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < offsets.len() {
        let end = (start + budget).min(offsets.len());
        ranges.push((offsets[start].0, offsets[end - 1].1));
        if end == offsets.len() {
            break;
        }
        start += stride;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::window_ranges;

    #[test]
    fn test_window_ranges_cover_every_token() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(
            window_ranges(&offsets, 2, 2),
            vec![(0, 5), (6, 11), (12, 14)]
        );
        assert_eq!(window_ranges(&offsets, 5, 5), vec![(0, 14)]);
        assert_eq!(window_ranges(&offsets, 8, 4), vec![(0, 14)]);
    }

    #[test]
    fn test_window_ranges_overlap_on_a_short_stride() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(window_ranges(&offsets, 4, 2), vec![(0, 11), (6, 14)]);
    }

    #[test]
    fn test_window_ranges_edges() {
        assert!(window_ranges(&[], 4, 2).is_empty());
        assert!(window_ranges(&[(0, 2)], 0, 1).is_empty());
        assert_eq!(window_ranges(&[(0, 2)], 1, 0), vec![(0, 2)]);
    }
}
